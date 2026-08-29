import ctypes
import hashlib
import json
import math
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Paddle/PIR compatibility
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

# TODO: add per-camera frame rate control
# TODO: Bent plates are harder to detect for model currently
# TODO: Plates that extend to multiple lines (like on motorcycles) are harder to detect for model currently (only flags as a plate, but doesnt read its number)

#Configs:
CONF_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 300
FRAME_SKIP_MAX = 29
WORKER_THREADS_NUMBER = 6
ENABLE_CAMERA_RECONNECT = True
RECONNECT_INTERVAL_SECONDS = 5.0
MAX_CONSECUTIVE_READ_FAILURES = 25

# OCR/tracking reuse controls
OCR_REUSE_SECONDS = 1.0
OCR_HIGH_CONF_MIN = 0.6
TRACK_MATCH_IOU = 0.45
OCR_MIN_DET_CONF = 0.40
OCR_MAX_CROP_WIDTH = 320
OCR_MIN_CROP_WIDTH = 200

# Legacy flush controls kept as no-op compatibility; capture is now worker-thread based.
ENABLE_FRAME_FLUSH = False
FLUSH_FRAMES_PER_CYCLE = 2
FLUSH_ONLY_WHEN_LAGGING = False
MAX_FRAME_AGE_MS = 250
MAX_CONSECUTIVE_FLUSH = 5
LOG_FLUSH_METRICS_EVERY_N = 100

LOG_EVERY_N_CYCLES = 30

CAMERAS = { #Apparently even the second set of numbers can change. (depends on the wifi i think)
    # "CAM_01": "http://10.200.197.70:8080/video",
    "CAM_02": "http://10.200.127.27:8080/video",
    # "CAM_03": "http://10.200.195.67:8080/video",
}

YOLO_MODEL = "model/plate_detector.pt"
OUTPUT_DIR = Path("multi_camera_output")
CROPS_DIR = OUTPUT_DIR / "crops"
FRAMES_DIR = OUTPUT_DIR / "frames"
JSON_PATH = OUTPUT_DIR / "events.json"

OUTPUT_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# P2 - format-aware Indian plate correction
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HR", "HP", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB", "AN", "TG", "BH"
}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}

class InvalidVideoFeedback(Exception):
    pass
class InvalidWorkerThreading(Exception):
    pass
# P1 - Non-blocking capture buffer
class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0

    def update(self, frame):
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get(self):
        with self._lock:
            return self._frame, self._frame_id

def capture_worker(camera_id: str, cap, buffer: LatestFrameBuffer, stop_event: threading.Event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        buffer.update(frame)

# P1 - OCR worker pool, one PaddleOCR instance per worker
class OCRWorkerPool:
    def __init__(self, num_workers=3, lang="en"):
        self.task_queue = queue.Queue(maxsize=64)
        self.result_queue = queue.Queue(maxsize=256)
        self._workers = []
        for _ in range(num_workers):
            ocr_instance = PaddleOCR(lang=lang, enable_mkldnn=False)
            worker = threading.Thread(target=self._worker_loop, args=(ocr_instance,), daemon=True)
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self, ocr_instance):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            track_id, camera_id, crop = task
            try:
                results = ocr_instance.predict(crop)
                text, conf = self._extract(results)
            except Exception:
                text, conf = "", 0.0
            self.result_queue.put((track_id, camera_id, text, conf))

    @staticmethod
    def _extract(ocr_results):
        for result in ocr_results or []:
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            if texts:
                return str(texts[0]).upper(), float(scores[0]) if scores else 0.0
        return "", 0.0

    def submit(self, track_id, camera_id, crop):
        try:
            self.task_queue.put_nowait((track_id, camera_id, crop))
        except queue.Full:
            pass

    def poll_results(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get_nowait())
        return results

    def stop(self):
        for _ in self._workers:
            self.task_queue.put_nowait(None)

# P3 - Crop normalization helpers
def pad_bbox(x1, y1, x2, y2, w, h, pad_frac=0.08):
    pad_x = int((x2 - x1) * pad_frac)
    pad_y = int((y2 - y1) * pad_frac)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
def normalize_plate_size(plate, min_width=OCR_MIN_CROP_WIDTH, max_width=OCR_MAX_CROP_WIDTH):
    h, w = plate.shape[:2]
    if w < min_width:
        scale = min_width / float(w)
        return cv2.resize(plate, (min_width, max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    if w > max_width:
        scale = max_width / float(w)
        return cv2.resize(plate, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return plate
def apply_crop_preprocessing(plate):
    if plate is None or plate.size == 0:
        return plate
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if len(plate.shape) == 3 else plate
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sharpen = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharpen = cv2.addWeighted(gray, 1.7, sharpen, -0.5, 0)
    return cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR) if len(plate.shape) == 3 else sharpen
def downscale_plate_if_needed(plate):
    h, w = plate.shape[:2]
    if w <= OCR_MAX_CROP_WIDTH:
        return plate
    scale = OCR_MAX_CROP_WIDTH / float(w)
    new_w = OCR_MAX_CROP_WIDTH
    new_h = max(1, int(h * scale))
    return cv2.resize(plate, (new_w, new_h), interpolation=cv2.INTER_AREA)
def correct_plate(raw_text: str) -> str:
    text = (raw_text or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not (9 <= len(text) <= 10):
        return text

    chars = list(text)
    for i in range(len(chars) - 4, len(chars)):
        chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
    for i in range(0, 2):
        chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])
    for i in range(2, 4):
        chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
    for i in range(4, len(chars) - 4):
        chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])

    corrected = "".join(chars)
    state = corrected[:2]
    if state not in INDIAN_STATE_CODES:
        best = min(INDIAN_STATE_CODES, key=lambda c: sum(a != b for a, b in zip(c, state)))
        if sum(a != b for a, b in zip(best, state)) <= 1:
            corrected = best + corrected[2:]
    return corrected

# P4 - multi-sample voting
def vote_plate_text(samples):
    if not samples:
        return ""
    target_len = Counter(len(text) for text, _ in samples).most_common(1)[0][0]
    filtered = [(text, conf) for text, conf in samples if len(text) == target_len]
    if not filtered:
        filtered = samples
    if not filtered:
        return ""
    return "".join(Counter(text[i] for text, _ in filtered).most_common(1)[0][0] for i in range(target_len))
def get_screen_size() -> tuple[int, int]:
    return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
def tile_windows(camera_ids, SCREEN_WIDTH, SCREEN_HEIGHT):
    if not camera_ids:
        return
    count = len(camera_ids)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    margin = 10
    available_w = max(1, SCREEN_WIDTH - (margin * 2))
    available_h = max(1, SCREEN_HEIGHT - (margin * 2))
    cell_w = max(1, available_w // cols)
    cell_h = max(1, available_h // rows)

    for idx, camera_id in enumerate(camera_ids):
        row, col = divmod(idx, cols)
        x = margin + col * cell_w
        y = margin + row * cell_h
        cv2.namedWindow(camera_id, cv2.WINDOW_NORMAL)
        cv2.moveWindow(camera_id, x, y)
        cv2.resizeWindow(camera_id, cell_w, cell_h)
def compute_capped_skip(camera_id: str, cycle_index: int, max_skip: int) -> int:
    now_ns = time.time_ns()
    seed = f"{camera_id}:{cycle_index}:{now_ns}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return 1 + (int.from_bytes(digest[:4], byteorder="big") % max_skip)
def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

def bootDependencies():
    print("Loading YOLO......")
    yolo = YOLO(YOLO_MODEL)
    print("YOLO Loaded")

    print("Loading PaddleOCR")
    paddleOCR = PaddleOCR(lang="en", enable_mkldnn=False)
    print("Models Loaded")

    return yolo, paddleOCR, {}, {}, {}, threading.Event(), ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
def bootCameras(cameraBuffers, videoCaptures,
                SCREEN_WIDTH, SCREEN_HEIGHT, stopEvent):
    dashes = "=============================="
    space = " "*7
    print(f"\n{dashes} \n{space}Starting Cameras{space} \n{dashes}")

    videoCapture = None
    workerThread = None
    connectedCameras = set()
    workers = {}
    disconnectEvents = {}
    failCounts = {}
    for camera_id, url in CAMERAS.items():
        disconnectEvents[camera_id] = threading.Event()
        if connectCamera(camera_id, url, videoCaptures, cameraBuffers, stopEvent, workers, failCounts, disconnectEvents):
            connectedCameras.add(camera_id)

    tile_windows(list(videoCaptures.keys()), SCREEN_WIDTH, SCREEN_HEIGHT)
    activeWindow = next(iter(videoCaptures), None)

    if videoCapture is None:
        raise InvalidVideoFeedback("ERROR: No cameras connected, cap is None") #TODO: Make this error more verbose
    if workerThread is None:
        raise InvalidWorkerThreading("ERROR: No worker threads started.") #TODO: Make this error more verbose
    print(f"\n{dashes}  Multi-Camera ANPR Booted  \n{dashes}")
    print("Press Q to stop\n")

    return (videoCapture, workerThread, activeWindow, cameraBuffers, videoCaptures,
            workers, failCounts, disconnectEvents, connectedCameras)

def connectCamera(cameraId, url, videoCapturesMap, cameraBuffersMap,
                  stopEvent, workers, failCountMap, disconnectEvents):
    print(f"[{cameraId}] Connecting...")
    videoCapture = cv2.VideoCapture(url)
    videoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not videoCapture.isOpened():
        print(f"[{cameraId}] ERROR: Could not connect")
        videoCapture.release()
        return False

    videoCapturesMap[cameraId] = videoCapture
    cameraBuffersMap[cameraId] = LatestFrameBuffer()
    failCountMap[cameraId] = 0

    worker = threading.Thread(
        target=getWorker,
        args=(cameraId, videoCapture, cameraBuffersMap[cameraId], stopEvent, disconnectEvents, failCountMap),
        daemon=True
    )
    worker.start()
    workers[cameraId] = worker

    cv2.namedWindow(cameraId, cv2.WINDOW_NORMAL)
    print(f"[{cameraId}] Connected")
    return True
def disconnectCamera(cameraId, videoCaptures, workers):
    cap = videoCaptures.pop(cameraId, None)
    if cap is not None:
        cap.release()
    workers.pop(cameraId, None)
    try:
        cv2.destroyWindow(cameraId)
    except cv2.error:
        pass
    print(f"[{cameraId}] Disconnected")
def getWorker(camera_id, cap, buffer, stop_event, disconnect_event, fail_counts):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            fail_counts[camera_id] = 0
            buffer.update(frame)
            continue

        fail_counts[camera_id] = fail_counts.get(camera_id, 0) + 1
        if fail_counts[camera_id] >= MAX_CONSECUTIVE_READ_FAILURES:
            disconnect_event.set()
            break
        time.sleep(0.05)

def tryConnectDeadCameras(captures, camera_buffers, stop_event, workers, fail_counts,
                          connected, disconnect_events, SCREEN_WIDTH, SCREEN_HEIGHT):
    for cam_id, url in CAMERAS.items():
        if cam_id not in captures:  # only disconnected/not connected
            disconnect_events[cam_id] = threading.Event()
            if connectCamera(cam_id, url, captures, camera_buffers, stop_event, workers, fail_counts, disconnect_events):
                connected.add(cam_id)
    tile_windows(list(captures.keys()), SCREEN_WIDTH, SCREEN_HEIGHT)
# def inputListener(stop_event, refresh_callback, refreshMsg):
#     while not stop_event.is_set():
#         if not msvcrt.kbhit():
#             time.sleep(0.05)
#             continue
#         key = msvcrt.getwch().lower()
#         if key != 'r':
#             time.sleep(0.05)
#         print(refreshMsg)
#         refresh_callback()
#         time.sleep(0.05)

def main():
    detector, ocr, captures, camera_buffers, camera_last_frame_id, stop_event, SCREEN_WIDTH, SCREEN_HEIGHT = bootDependencies()
    cap, worker, ACTIVE_WINDOW, camera_buffers, captures, workers, fail_counts, disconnect_events, connectedCameras = bootCameras(camera_buffers, captures, SCREEN_WIDTH, SCREEN_HEIGHT, stop_event)

    events = []
    plate_counter = 0
    ocr_pool = OCRWorkerPool(num_workers=WORKER_THREADS_NUMBER, lang="en")

    onFrameVsCamera = {camera_id: 0 for camera_id in captures}
    per_camera_skip = {camera_id: 1 for camera_id in captures}
    per_camera_cycle = {camera_id: 0 for camera_id in captures}

    cached_detections = {cam_id: [] for cam_id in captures}
    ocr_track_cache = {cam_id: [] for cam_id in captures}
    ocr_vote_cache = {cam_id: {} for cam_id in captures}
    camera_metrics = {
        cam_id: {
            "frames_read": 0,
            "frames_processed": 0,
            "frames_flushed": 0,
            "ocr_calls": 0,
            "ocr_skips": 0,
            "consecutive_flushes": 0,
        }
        for cam_id in captures
    }

    cv2.namedWindow("CONTROL", cv2.WINDOW_NORMAL)
    cv2.moveWindow("CONTROL", 0, 0)

    try:
        lastReconnectAttemptTS = time.time()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQ pressed. Stopping...")
                break
            if key == ord("r"):
                print("R pressed: Attempting to reconnect dead cameras...")
                tryConnectDeadCameras(captures, camera_buffers, stop_event, workers,
                                      fail_counts, connectedCameras, disconnect_events, SCREEN_WIDTH, SCREEN_HEIGHT)
                tile_windows(list(captures.keys()), SCREEN_WIDTH, SCREEN_HEIGHT)

            now = time.time()
            if ENABLE_CAMERA_RECONNECT and (now - lastReconnectAttemptTS) >= RECONNECT_INTERVAL_SECONDS:
                tryConnectDeadCameras(captures, camera_buffers, stop_event, workers,
                                      fail_counts, connectedCameras, disconnect_events, SCREEN_WIDTH, SCREEN_HEIGHT)
                lastReconnectAttemptTS = now

            dead_cameras = [cam_id for cam_id, evt in disconnect_events.items() if evt.is_set() and cam_id in captures]
            for cam_id in dead_cameras:
                disconnectCamera(cam_id, captures, workers)
                camera_buffers.pop(cam_id, None)
                camera_last_frame_id.pop(cam_id, None)
                onFrameVsCamera.pop(cam_id, None)
                per_camera_skip.pop(cam_id, None)
                per_camera_cycle.pop(cam_id, None)
                cached_detections.pop(cam_id, None)
                ocr_track_cache.pop(cam_id, None)
                ocr_vote_cache.pop(cam_id, None)
                camera_metrics.pop(cam_id, None)

            for camera_id in list(captures.keys()):
                cap = captures.get(camera_id)
                if cap is None:
                    continue

                buffer = camera_buffers[camera_id]
                frame, frame_id = buffer.get()
                if frame is None:
                    continue
                if frame_id == camera_last_frame_id.get(camera_id, -1):
                    continue
                camera_last_frame_id[camera_id] = frame_id
                camera_metrics[camera_id]["frames_read"] += 1

                if onFrameVsCamera[camera_id] >= per_camera_skip[camera_id]:
                    onFrameVsCamera[camera_id] = 0
                    per_camera_cycle[camera_id] += 1
                    per_camera_skip[camera_id] = compute_capped_skip(
                        camera_id, per_camera_cycle[camera_id], FRAME_SKIP_MAX
                    )

                    current_detections = []
                    next_tracks = []
                    now_ts = time.time()

                    results = detector.predict(
                        source=frame,
                        conf=CONF_THRESHOLD,
                        imgsz=YOLO_IMAGE_SIZE,
                        iou=0.45,
                        verbose=False
                    )
                    result = results[0]

                    detection_count = len(result.boxes) if result.boxes is not None else 0
                    prev_tracks = ocr_track_cache[camera_id]

                    if detection_count > 0 and per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                        print(f"\n[{camera_id}] Detections: {detection_count}")

                    for box in result.boxes:
                        detection_confidence = float(box.conf[0].cpu().item())
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        h, w = frame.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        plate = frame[y1:y2, x1:x2]
                        if plate.size == 0:
                            continue

                        matched_track = None
                        best_iou = 0.0
                        for track in prev_tracks:
                            current_iou = bbox_iou((x1, y1, x2, y2), track["bbox"])
                            if current_iou > best_iou:
                                best_iou = current_iou
                                matched_track = track

                        plate_text = ""
                        ocr_confidence = 0.0
                        ocr_job_id = None

                        track_age = 0.0
                        if matched_track is not None:
                            track_age = now_ts - matched_track["first_seen_ts"]

                        can_reuse_high_conf = (
                            matched_track is not None
                            and best_iou >= TRACK_MATCH_IOU
                            and matched_track["ocr_conf"] >= OCR_HIGH_CONF_MIN
                            and track_age >= OCR_REUSE_SECONDS
                            and bool(matched_track["text"])
                        )

                        if can_reuse_high_conf:
                            plate_text = matched_track["text"]
                            ocr_confidence = matched_track["ocr_conf"]
                            camera_metrics[camera_id]["ocr_skips"] += 1
                        elif detection_confidence < OCR_MIN_DET_CONF:
                            plate_text = ""
                            ocr_confidence = 0.0
                        else:
                            x1_pad, y1_pad, x2_pad, y2_pad = pad_bbox(x1, y1, x2, y2, w, h, pad_frac=0.08)
                            plate = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            if plate.size == 0:
                                continue
                            plate = normalize_plate_size(apply_crop_preprocessing(plate))
                            plate = downscale_plate_if_needed(plate)
                            ocr_job_id = f"{camera_id}:{time.time_ns()}"
                            ocr_pool.submit(ocr_job_id, camera_id, plate)
                            camera_metrics[camera_id]["ocr_calls"] += 1

                        plate_text = re.sub(r"[^A-Z0-9]", "", plate_text)
                        now = datetime.now()
                        timestamp = now.isoformat(timespec="milliseconds")

                        det_entry = {
                            "bbox": (x1, y1, x2, y2),
                            "text": plate_text,
                            "det_conf": detection_confidence,
                            "ocr_conf": ocr_confidence,
                            "ocr_job_id": ocr_job_id,
                            "ts": timestamp,
                        }
                        current_detections.append(det_entry)

                        first_seen_ts = now_ts
                        if matched_track is not None and best_iou >= TRACK_MATCH_IOU:
                            first_seen_ts = matched_track["first_seen_ts"]

                        next_tracks.append({
                            "bbox": (x1, y1, x2, y2),
                            "text": plate_text,
                            "ocr_conf": ocr_confidence,
                            "ocr_job_id": ocr_job_id,
                            "first_seen_ts": first_seen_ts,
                            "last_seen_ts": now_ts,
                            "vote_samples": [(plate_text, ocr_confidence)] if plate_text else [],
                        })

                    for job_id, job_camera, ocr_text, ocr_conf in ocr_pool.poll_results():
                        if job_camera != camera_id:
                            continue
                        refined_text = correct_plate(ocr_text) if ocr_text else ""
                        refined_conf = float(ocr_conf)
                        for det in current_detections:
                            if det.get("ocr_job_id") == job_id:
                                det["text"] = refined_text
                                det["ocr_conf"] = refined_conf
                        for track in next_tracks:
                            if track.get("ocr_job_id") == job_id:
                                track["text"] = refined_text
                                track["ocr_conf"] = refined_conf
                                track["vote_samples"] = (track.get("vote_samples") or []) + [(refined_text, refined_conf)]
                                if len(track["vote_samples"]) > 5:
                                    track["vote_samples"] = track["vote_samples"][-5:]
                                track["text"] = vote_plate_text(track["vote_samples"]) if track["vote_samples"] else refined_text

                    for det in current_detections:
                        text = det["text"]
                        if not text:
                            continue
                        det["text"] = re.sub(r"[^A-Z0-9]", "", text)
                        det["ocr_conf"] = float(det.get("ocr_conf", 0.0))
                        if det["text"]:
                            plate_counter += 1
                            crop_name = f"{camera_id}_{plate_counter:05d}_{det['text']}.jpg"
                            crop_path = CROPS_DIR / crop_name
                            plate_crop = frame[int(det["bbox"][1]):int(det["bbox"][3]), int(det["bbox"][0]):int(det["bbox"][2])]
                            if plate_crop.size > 0:
                                cv2.imwrite(str(crop_path), plate_crop)
                            event = {
                                "camera_id": camera_id,
                                "plate": det["text"],
                                "timestamp": det["ts"],
                                "detection_confidence": float(det["det_conf"]),
                                "ocr_confidence": float(det["ocr_conf"]),
                                "bounding_box": {"x1": int(det["bbox"][0]), "y1": int(det["bbox"][1]), "x2": int(det["bbox"][2]), "y2": int(det["bbox"][3])},
                                "crop": str(crop_path),
                            }
                            events.append(event)
                            if per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                                print(f"[{det['ts']}] [{camera_id}] {det['text']} | DET {det['det_conf']:.3f} | OCR {det['ocr_conf']:.3f}")

                    cached_detections[camera_id] = current_detections
                    ocr_track_cache[camera_id] = next_tracks
                    camera_metrics[camera_id]["frames_processed"] += 1

                    if per_camera_cycle[camera_id] % LOG_FLUSH_METRICS_EVERY_N == 0:
                        m = camera_metrics[camera_id]
                        print(
                            f"[{camera_id}] metrics read={m['frames_read']} processed={m['frames_processed']} "
                            f"flushed={m['frames_flushed']} ocr_calls={m['ocr_calls']} ocr_skips={m['ocr_skips']}"
                        )
                else:
                    onFrameVsCamera[camera_id] += 1

                for det in cached_detections[camera_id]:
                    x1, y1, x2, y2 = det["bbox"]
                    text = det["text"]
                    detection_confidence = det["det_conf"]
                    ocr_confidence = det["ocr_conf"]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    if text:
                        label = f"{text} | {detection_confidence:.2f} | OCR {ocr_confidence:.2f}"
                    else:
                        label = f"PLATE | {detection_confidence:.2f}"
                    cv2.putText(frame, label, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(frame, camera_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, current_time, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(camera_id, frame)

    finally:
        stop_event.set()
        ocr_pool.stop()
        print("\nStopping cameras...")
        for camera_id, cap in captures.items():
            cap.release()
            print(f"[{camera_id}] Released")
        cv2.destroyAllWindows()

    output = {
        "system": "Multi-Camera ANPR",
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "camera_count": len(captures),
        "total_events": len(events),
        "events": events
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print()
    print("==============================")
    print("ANPR Finished")
    print("==============================")
    print(f"Total events: {len(events)}")
    print(f"JSON saved: {JSON_PATH}")

if __name__ == "__main__":
    main()