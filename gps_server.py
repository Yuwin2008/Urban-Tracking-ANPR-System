from flask import Flask, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)

GPS_FILE = "camera_locations.json"
HISTORY_FILE = "camera_history.json"

SERVER_HOST = os.getenv("GPS_SERVER_HOST", "0.0.0.0")   # bind address
SERVER_PORT = int(os.getenv("GPS_SERVER_PORT", "5000"))
PUBLIC_BASE_URL = os.getenv(
    "GPS_PUBLIC_BASE_URL",
    f"http://{SERVER_HOST}:{SERVER_PORT}" if SERVER_HOST != "0.0.0.0"
    else f"http://localhost:{SERVER_PORT}"
)

@app.after_request
def add_cors_headers(response):
    # Allows the dashboard (index.html), which is opened from a different
    # origin (file:// or another port), to fetch /gps and /gps/history.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default
def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

@app.route("/gps", methods=["POST"])
def receive_gps():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON received"
        }), 400

    camera_id = data.get("camera_id", "UNKNOWN")
    data["server_received"] = datetime.now().isoformat()
    locations = load_json(GPS_FILE, {})
    locations[camera_id] = data
    save_json(GPS_FILE, locations)

    history = load_json(HISTORY_FILE, {})
    if camera_id not in history:
        history[camera_id] = []
    history[camera_id].append(data)
    save_json(HISTORY_FILE, history)

    print(
        f"[{camera_id}] "
        f"Lat: {data.get('latitude')} | "
        f"Lon: {data.get('longitude')} | "
        f"Accuracy: {data.get('accuracy')}m | "
        f"Time: {data.get('timestamp')}"
    )

    return jsonify({
        "status": "ok",
        "camera_id": camera_id
    })
@app.route("/gps", methods=["GET"])
def get_gps():
    return jsonify(load_json(GPS_FILE, {}))


@app.route("/gps/history", methods=["GET"])
def get_history():
    return jsonify(
        load_json(HISTORY_FILE, {})
    )
@app.route("/gps/history/<camera_id>", methods=["GET"])
def get_camera_history(camera_id):
    history = load_json(HISTORY_FILE, {})
    return jsonify(
        history.get(camera_id, [])
    )
def main():
    print("=" * 50)
    print("UrbanTracker GPS Server")
    print("=" * 50)

    print("Latest GPS:")
    print(f"{PUBLIC_BASE_URL}/gps")

    print("All GPS history:")
    print(f"{PUBLIC_BASE_URL}/gps/history")

    # print("CAM_01 history:")
    # print(f"{PUBLIC_BASE_URL}/gps/history/CAM_01")

    print()
    print("Server running...")
    print()

    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)

if __name__ == "__main__":
    main()