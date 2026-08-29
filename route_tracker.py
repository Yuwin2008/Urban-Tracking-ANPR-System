"""
route_tracker.py
=====================================================================
Reads multi_camera_output/events.json (written continuously by
anpr_multi_camera.py) and camera_config.json, reconstructs each
plate's route across cameras, and writes:
  - map_ouput/routes.json  (consumed live by the website)
  - map_ouput/map.html     (a standalone folium map, for manual viewing)

Runs in a loop: it watches events.json's modified time and rebuilds
routes.json a couple of seconds after new events land, so the
website's Trajectory Tracking page stays in sync with the camera
script without you having to re-run this by hand.

Run:
    python route_tracker.py
=====================================================================
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import folium

EVENTS_FILE = Path("multi_camera_output/events.json")
CAMERA_FILE = Path("camera_config.json")

OUTPUT_DIR = Path("map_ouput")
ROUTES_FILE = OUTPUT_DIR / "routes.json"
MAP_FILE = OUTPUT_DIR / "map.html"

REBUILD_INTERVAL_SECONDS = 3.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_routes(cameras, events):
    """Groups events by plate, in camera-visit order, and works out the
    transitions (travel time / distance / speed) between consecutive
    camera sightings for each plate."""
    vehicles = defaultdict(list)

    for event in events:
        plate = event.get("plate")
        camera_id = event.get("camera_id")
        timestamp = event.get("timestamp")

        if not plate:
            continue
        if camera_id not in cameras:
            continue
        if not timestamp:
            continue

        vehicles[plate].append(event)

    routes = []

    for plate, plate_events in vehicles.items():
        plate_events.sort(key=lambda x: x["timestamp"])

        # Collapse consecutive detections at the same camera into one
        # visit, so a car sitting in front of one camera for a while
        # doesn't get counted as bouncing back and forth.
        visits = []
        for event in plate_events:
            if visits and visits[-1]["camera_id"] == event["camera_id"]:
                continue
            visits.append(event)

        if len(visits) < 2:
            continue  # need at least 2 different cameras to have a "route"

        route_points = []
        for event in visits:
            camera_id = event["camera_id"]
            camera = cameras[camera_id]
            route_points.append({
                "camera_id": camera_id,
                "camera_name": camera["name"],
                "latitude": camera["latitude"],
                "longitude": camera["longitude"],
                "timestamp": event["timestamp"],
                "plate": plate,
            })

        transitions = []
        for i in range(len(route_points) - 1):
            current = route_points[i]
            next_point = route_points[i + 1]

            t1 = datetime.fromisoformat(current["timestamp"])
            t2 = datetime.fromisoformat(next_point["timestamp"])
            travel_time = (t2 - t1).total_seconds()

            distance = haversine(
                current["latitude"], current["longitude"],
                next_point["latitude"], next_point["longitude"],
            )

            speed = (distance / (travel_time / 3600)) if travel_time > 0 else None

            transitions.append({
                "from_camera": current["camera_id"],
                "to_camera": next_point["camera_id"],
                "start_time": current["timestamp"],
                "end_time": next_point["timestamp"],
                "travel_time_seconds": travel_time,
                "distance_km": round(distance, 3),
                "estimated_speed_kmph": round(speed, 2) if speed is not None else None,
            })

        routes.append({
            "plate": plate,
            "camera_count": len(route_points),
            "route": route_points,
            "transitions": transitions,
        })

    return routes


def save_routes(routes):
    route_data = {"total_vehicles": len(routes), "vehicles": routes}
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp_path = ROUTES_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(route_data, f, indent=4)
    tmp_path.replace(ROUTES_FILE)


def save_map(cameras, routes):
    if cameras:
        first_camera = next(iter(cameras.values()))
        map_center = [first_camera["latitude"], first_camera["longitude"]]
    else:
        map_center = [20.5937, 78.9629]  # fallback: center of India

    m = folium.Map(location=map_center, zoom_start=14)

    for camera_id, camera in cameras.items():
        folium.Marker(
            location=[camera["latitude"], camera["longitude"]],
            popup=f"<b>{camera_id}</b><br>{camera['name']}",
            tooltip=camera_id,
            icon=folium.Icon(icon="camera", prefix="fa"),
        ).add_to(m)

    for vehicle in routes:
        plate = vehicle["plate"]
        points = vehicle["route"]
        coordinates = [[p["latitude"], p["longitude"]] for p in points]

        folium.PolyLine(
            coordinates,
            tooltip=f"Vehicle: {plate}",
            popup=f"<b>Vehicle:</b> {plate}<br><b>Cameras:</b> {len(points)}",
            weight=5,
        ).add_to(m)

        for point in points:
            popup_html = (
                f"<b>Vehicle:</b> {plate}<br>"
                f"<b>Camera:</b> {point['camera_id']}<br>"
                f"<b>Time:</b> {point['timestamp']}"
            )
            folium.CircleMarker(
                location=[point["latitude"], point["longitude"]],
                radius=7,
                popup=popup_html,
                tooltip=plate,
                fill=True,
            ).add_to(m)

    OUTPUT_DIR.mkdir(exist_ok=True)
    m.save(str(MAP_FILE))


def run_once():
    with open(CAMERA_FILE, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    routes = build_routes(cameras, events)

    save_routes(routes)
    save_map(cameras, routes)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"{len(events)} events -> {len(routes)} vehicle route(s). "
          f"Saved to {ROUTES_FILE} and {MAP_FILE}.")


if __name__ == "__main__":
    print("Watching", EVENTS_FILE, "- rebuilding routes whenever it changes.")
    print("Press Ctrl+C to stop.\n")

    last_mtime = None
    while True:
        try:
            if not EVENTS_FILE.exists():
                print(f"Waiting for {EVENTS_FILE} to exist "
                      f"(start anpr_multi_camera.py first)...")
                time.sleep(REBUILD_INTERVAL_SECONDS)
                continue

            mtime = EVENTS_FILE.stat().st_mtime
            if mtime != last_mtime:
                run_once()
                last_mtime = mtime
        except (json.JSONDecodeError, KeyError):
            # events.json can be mid-write for an instant; just retry next cycle
            pass
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        time.sleep(REBUILD_INTERVAL_SECONDS)
