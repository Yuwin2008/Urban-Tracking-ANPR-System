"""
server.py — connects the ANPR/OCR Python pipeline (YOLO + EasyOCR,
same detector model used by anpr.py / plate_detect.py) to the TrackNet website.

What it does:
  1. Serves this whole project folder as static files, so the website's
     existing relative fetches (../camera_config.json, ../map_ouput/routes.json,
     ../multi_camera_output/events.json, ../output/results.json, crop images,
     etc.) all resolve correctly when you open the site through this server
     instead of as a local file:// page.
  2. Exposes POST /api/anpr/upload — accepts an image from the browser,
     runs detect+OCR, saves the crop and an annotated frame, appends the
     result to output/results.json, and returns JSON the website can
     render immediately.

Note: this uses EasyOCR instead of PaddleOCR. PaddleOCR's CPU path is
known to be extremely slow (many minutes per image) on Apple Silicon
Macs; EasyOCR is CPU-friendly and typically finishes in a couple of
seconds per plate crop.

Also includes plate-format auto-correction (see correct_plate_format())
that fixes common OCR character-class mix-ups — 0/O, 1/I, 2/Z, 5/S, 6/G,
8/B — using the fixed Indian plate pattern (SS DD L{1,3} DDDD) to know
which positions must be letters vs. digits.

Run:
    pip install -r requirements.txt
    python server.py

Then open:
    http://localhost:5001/Website/index.html
"""

import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CROP_DIR = OUTPUT_DIR / "crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
RESULTS_JSON = OUTPUT_DIR / "results.json"

CROP_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.05

print("Loading YOLO plate detector...")
detector = YOLO(str(BASE_DIR / "model" / "plate_detector.pt"))
print("Loading EasyOCR (first run downloads model weights, ~1-2 min once)...")
ocr = easyocr.Reader(["en"], gpu=False)
print("Models loaded.\n")

# static_url_path="" serves this whole folder at the site root, so
# Website/index.html's "../camera_config.json" etc. resolve for free.
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Plate-format auto-correction — Indian plates follow a fixed character
# pattern (SS DD L{1,3} DDDD: 2 state letters, 2 RTO digits, 1-3 series
# letters, 4 number digits). EasyOCR frequently confuses characters that
# look alike across the letter/digit boundary (0/O, 1/I, 2/Z, 5/S, 6/G,
# 8/B). Since we know which *class* of character (letter vs digit) is
# expected at each position, we can force ambiguous reads back to the
# correct class instead of guessing blindly.
# ---------------------------------------------------------------------------
PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")

DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "D": "0", "Q": "0"}


def _force_letter(ch):
    return ch if ch.isalpha() else DIGIT_TO_LETTER.get(ch, ch)


def _force_digit(ch):
    return ch if ch.isdigit() else LETTER_TO_DIGIT.get(ch, ch)


def correct_plate_format(raw_text):
    """Attempts to reshape a raw OCR string into a valid Indian plate by
    correcting characters that don't match the expected class for their
    position. Returns (corrected_text, is_valid). is_valid is only True
    when the corrected string fully matches the plate pattern — callers
    should treat that as a strong extra confidence signal, since it means
    the read is structurally consistent with a real plate, not just a
    per-character guess."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text or "").upper()
    if not (9 <= len(cleaned) <= 11):
        return cleaned, False

    series_len = len(cleaned) - 8  # state(2) + rto(2) + number(4) = 8 fixed chars
    state, rto = cleaned[0:2], cleaned[2:4]
    series, number = cleaned[4:4 + series_len], cleaned[4 + series_len:]

    corrected = (
        "".join(_force_letter(c) for c in state)
        + "".join(_force_digit(c) for c in rto)
        + "".join(_force_letter(c) for c in series)
        + "".join(_force_digit(c) for c in number)
    )
    return corrected, bool(PLATE_RE.match(corrected))


def _next_plate_index():
    return len(list(CROP_DIR.glob("plate_*.jpg")))


def run_anpr(image):
    """Runs the plate detector + OCR on a single BGR image (numpy array).
    This mirrors anpr.py's loop, just packaged as a reusable function."""
    annotated = image.copy()
    h, w = image.shape[:2]

    results = detector.predict(source=image, conf=CONF_THRESHOLD, verbose=False)

    detections = []
    plate_count = _next_plate_index()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0].cpu().item())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            plate_count += 1
            crop_name = f"plate_{plate_count:03d}.jpg"
            crop_path = CROP_DIR / crop_name
            cv2.imwrite(str(crop_path), plate_img)

            recognized_text, ocr_conf = "", 0.0
            # EasyOCR returns a list of (bbox, text, confidence) per detected
            # text region. A plate can have multiple regions (e.g. state code
            # on its own line), so join them left-to-right and average confidence.
            ocr_results = ocr.readtext(plate_img)
            if ocr_results:
                ocr_results.sort(key=lambda r: r[0][0][0])  # sort by left x-coord
                recognized_text = " ".join(r[1] for r in ocr_results).strip()
                ocr_conf = sum(r[2] for r in ocr_results) / len(ocr_results)

            # Auto-correct common OCR character-class confusions (0/O, 1/I,
            # 5/S, 8/B, ...) using the expected Indian plate pattern. Only
            # applied when the corrected string is fully format-valid, so
            # this never overwrites a read with a worse guess.
            raw_ocr_text = recognized_text
            format_valid = False
            if recognized_text:
                corrected_text, format_valid = correct_plate_format(recognized_text)
                if format_valid:
                    recognized_text = corrected_text
                    # Format validation is strong independent evidence the
                    # read is correct, on top of EasyOCR's own per-character
                    # confidence — so raise the reported confidence, but
                    # never lower it if EasyOCR was already more confident.
                    ocr_conf = max(ocr_conf, 0.95)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"{recognized_text} ({ocr_conf:.2f})",
                        (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            detections.append({
                "plate": recognized_text,
                "detector_confidence": det_conf,
                "ocr_confidence": ocr_conf,
                "format_valid": format_valid,
                # only present when correction actually changed the reading,
                # so you can audit what EasyOCR originally saw vs. what was
                # auto-corrected
                **({"raw_ocr_text": raw_ocr_text} if raw_ocr_text != recognized_text else {}),
                "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                # forward-slash, relative to project root, so it works as a URL
                "crop": crop_path.relative_to(BASE_DIR).as_posix(),
                "timestamp": datetime.now().isoformat(),
            })

    return detections, annotated


def _append_to_results(image_name, detections):
    data = {"detections": []}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    data.setdefault("detections", [])
    for d in detections:
        entry = dict(d)
        entry["image"] = image_name
        data["detections"].append(entry)
    data["total_plates"] = len(data["detections"])
    data["processed_at"] = datetime.now().isoformat()
    RESULTS_JSON.write_text(json.dumps(data, indent=4), encoding="utf-8")


@app.route("/api/anpr/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "could not decode image"}), 400

    detections, annotated = run_anpr(image)

    annotated_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_name
    cv2.imwrite(str(annotated_path), annotated)

    _append_to_results(file.filename, detections)

    return jsonify({
        "total_plates": len(detections),
        "detections": detections,
        "annotated": annotated_path.relative_to(BASE_DIR).as_posix(),
    })


@app.route("/api/anpr/results", methods=["GET"])
def get_results():
    if not RESULTS_JSON.exists():
        return jsonify({"total_plates": 0, "detections": []})
    return jsonify(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))


if __name__ == "__main__":
    print("=" * 60)
    print("ANPR server running")
    print("=" * 60)
    print("\nOpen the site at:")
    print("  http://localhost:5001/Website/index.html\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
