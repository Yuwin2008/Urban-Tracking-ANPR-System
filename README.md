#  Urban Tracking ANPR System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111F68?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-OCR-00A67E)](https://github.com/PaddlePaddle/PaddleOCR)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Folium](https://img.shields.io/badge/Folium-Maps-77B829)](https://python-visualization.github.io/folium/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Yuwin2008/Urban-Tracking-ANPR-System?style=flat&logo=github)](https://github.com/Yuwin2008/Urban-Tracking-ANPR-System/stargazers)

### AI-Powered Multi-Camera Automatic Number Plate Recognition & Vehicle Trajectory Tracking

An intelligent **urban traffic monitoring system** that combines **Automatic Number Plate Recognition (ANPR)**, **multi-camera vehicle tracking**, **OCR**, **GPS-based camera localization**, and **trajectory reconstruction** to track vehicles across a network of cameras.

The system is designed to transform independent camera feeds into a unified view of vehicle movement across an urban environment.

---

##  Overview

Traditional ANPR systems generally answer a simple question:

> **"What vehicle is visible in this camera?"**

This project goes further:

> **"Where has this vehicle been, which cameras detected it, and what route did it take?"**

The system processes vehicle/license-plate observations from multiple cameras, associates observations with license plate identities, attaches geographic information to camera locations, and reconstructs the vehicle's movement over time.

The resulting data can be visualized as an interactive trajectory on a map.

---

##  Key Features

*  **Automatic Number Plate Detection**
*  **License Plate OCR**
*  **Multi-Camera Video Processing**
*  **Cross-Camera Vehicle Identification**
*  **GPS-Based Camera Localization**
*  **Vehicle Route Reconstruction**
*  **Interactive Trajectory Maps**
*  **Time-Ordered Vehicle Movement**
*  **Exact and Fuzzy Plate Searching**
*   **Web-Based Dashboard**
*  **Flask API**
*  **JSON-Based Event and Detection Storage**

---

#  System Architecture

```text
                    ┌───────────────────────┐
                    │    Camera Network     │
                    │                       │
                    │ CAM 01 │ CAM 02 │ ... │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Video Acquisition   │
                    │      OpenCV            │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Plate Detection     │
                    │       YOLO             │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Plate Cropping      │
                    │   + Preprocessing      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        OCR            │
                    │ PaddleOCR / EasyOCR   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Detection Events      │
                    │ Plate + Camera + Time │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  ANPR Data Store      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Trajectory Engine     │
                    │                       │
                    │ • Matching            │
                    │ • Deduplication       │
                    │ • Transitions         │
                    │ • Anomalies           │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Route Visualization │
                    │      Folium Maps      │
                    └───────────────────────┘
```

---

#  AI Pipeline

## 1. License Plate Detection

The system uses a trained **YOLO-based plate detector** to locate license plates in incoming images or video frames.

```text
Video Frame
     ↓
YOLO Detector
     ↓
Plate Bounding Box
     ↓
Plate Crop
```

The trained detector is stored under:

```text
model/
└── plate_detector.pt
```

---

## 2. Optical Character Recognition

After detecting a plate, the system crops the detected region and passes it to an OCR engine to extract the plate characters.

The project currently contains implementations using:

* **PaddleOCR**
* **EasyOCR**

This allows the detection and recognition stages to remain modular.

---

## 3. Multi-Camera Processing

The multi-camera engine can connect to several network video streams simultaneously.

Example configuration:

```python
CAMERAS = {
    "CAM_01": "http://CAMERA_01/video",
    "CAM_02": "http://CAMERA_02/video",
    "CAM_03": "http://CAMERA_03/video",
}
```

Each camera is treated as an independent observation source.

The system records:

* Camera ID
* Plate number
* Detection confidence
* Timestamp
* Image/frame information
* Camera location

The multi-camera implementation also contains mechanisms for frame skipping, OCR reuse, detection caching, and frame flushing to help control latency during live processing.

---

#  Vehicle Trajectory Reconstruction

The core idea of the project is to transform individual ANPR detections into a **vehicle trajectory**.

For example:

```text
CAM_01
  ↓
CAM_04
  ↓
CAM_07
  ↓
CAM_09
```

Instead of treating these as four unrelated detections, the trajectory engine can reconstruct them as one vehicle journey.

The trajectory reconstruction engine supports:

* Exact plate queries
* Fuzzy plate matching
* Time-window filtering
* Detection deduplication
* Camera metadata attachment
* Camera-to-camera transitions
* Anomaly detection
* Geographic trajectory reconstruction

---

#  Map Visualization

Vehicle trajectories can be visualized using **Folium**.

A generated trajectory contains:

```text
Vehicle
   │
   ├── Camera 01
   │      └── Timestamp
   │
   ├── Camera 04
   │      └── Timestamp
   │
   └── Camera 07
          └── Timestamp
```

The system can render these points as a connected route on an interactive map.

Each point can contain information such as:

* Camera name
* Camera ID
* Timestamp
* Detection confidence
* Anomaly information

The map also provides trajectory-level information such as total distance, duration, and detected anomalies.

---

#  Anomaly Detection

The trajectory engine can annotate unusual transitions or observations.

This enables future extensions such as:

* Unexpected camera transitions
* Impossible movement patterns
* Suspicious timing between cameras
* Duplicate observations
* Inconsistent plate observations

The goal is to provide a foundation for **urban traffic intelligence**, rather than simply reading license plates.

---

#  Web Dashboard

The repository contains a web interface for interacting with the system.

```text
Website/
├── Dashboard
├── Camera information
├── Vehicle information
├── ANPR results
└── Trajectory visualization
```

The project also includes:

```text
vehicle-login.html
index.html
```

for the vehicle-facing/dashboard interface.

---

#  Backend API

A Flask-based backend is included for connecting the AI pipeline with the web interface.

The server provides the infrastructure for:

* Receiving image requests
* Running ANPR
* Returning recognition results
* Serving the web interface
* Accessing generated output data

The Flask server uses CORS to allow communication between the frontend and backend.

---

#  Project Structure

```text
Urban-Tracking-ANPR-System/
│
├── Website/
│   └── Web dashboard files
│
├── camera/
│   └── Camera-related components
│
├── detector/
│   └── Detection resources
│
├── model/
│   └── plate_detector.pt
│
├── multi_camera_output/
│   ├── crops/
│   ├── frames/
│   └── events.json
│
├── output/
│   ├── crops/
│   ├── annotated/
│   └── results.json
│
├── map_ouput/
│   └── Generated trajectory maps
│
├── anpr.py
│   └── Single-image ANPR pipeline
│
├── anpr_multi_camera.py
│   └── Multi-camera ANPR processing
│
├── camera_config.json
│   └── Camera metadata/configuration
│
├── camera_manager.py
│   └── Camera management
│
├── gps_server.py
│   └── GPS/location services
│
├── plate_detect.py
│   └── License plate detection
│
├── ocr_test.py
│   └── OCR testing
│
├── route_tracker.py
│   └── Route generation and map visualization
│
├── trajectory_engine.py
│   └── Trajectory reconstruction engine
│
├── bridge.py
│   └── System integration/communication
│
├── server.py
│   └── Flask backend
│
├── index.html
│   └── Main dashboard
│
├── vehicle-login.html
│   └── Vehicle login interface
│
├── yolo11n.pt
│   └── YOLO model
│
├── requirements.txt
└── README.md
```

The repository currently contains the major components shown above, including the detector, camera, model, output, mapping, web, and trajectory modules.

---

#  Technologies Used

| Technology              | Purpose                        |
| ----------------------- | ------------------------------ |
| **Python**              | Core development               |
| **YOLO / Ultralytics**  | License plate detection        |
| **PaddleOCR**           | OCR / text recognition         |
| **EasyOCR**             | Alternative OCR pipeline       |
| **OpenCV**              | Video processing               |
| **Flask**               | Backend/API                    |
| **Flask-CORS**          | Frontend/backend communication |
| **Folium**              | Interactive maps               |
| **NumPy**               | Numerical processing           |
| **JSON**                | Detection/event storage        |
| **HTML/CSS/JavaScript** | Web dashboard                  |

The current `requirements.txt` specifies NumPy, OpenCV, Ultralytics, EasyOCR, Flask, and Flask-CORS as core dependencies.

---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/Yuwin2008/Urban-Tracking-ANPR-System.git
cd Urban-Tracking-ANPR-System
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Configure Cameras

Edit:

```text
camera_config.json
```

or configure the camera streams in:

```text
anpr_multi_camera.py
```

Example:

```python
CAMERAS = {
    "CAM_01": "http://CAMERA_IP_1:8080/video",
    "CAM_02": "http://CAMERA_IP_2:8080/video",
    "CAM_03": "http://CAMERA_IP_3:8080/video"
}
```

The cameras should be reachable from the machine running the ANPR server.

---

#  Running ANPR

## Single Image

Run:

```bash
python anpr.py
```

The pipeline loads the YOLO plate detector, detects plates, crops them, and performs OCR.

---

## Multi-Camera ANPR

Run:

```bash
python anpr_multi_camera.py
```

The system connects to the configured camera streams and processes them concurrently.

Press:

```text
Q
```

to stop the live processing.

Detected events are written to the multi-camera output directory.

---

#  Running the Web Server

Start the Flask backend:

```bash
python server.py
```

Then open:

```text
http://localhost:5001/
```

The backend serves the project website and exposes the ANPR processing functionality to the frontend.

---

#  Route Tracking

After collecting ANPR observations, the route tracker can reconstruct vehicle routes.

```bash
python route_tracker.py
```

The generated route data contains information such as:

```json
{
    "total_vehicles": 1,
    "vehicles": [
        {
            "plate": "XX00XX0000",
            "route": [],
            "transitions": []
        }
    ]
}
```

The route tracker can then generate an interactive Folium map containing camera locations and vehicle routes.

---

#  Querying Vehicle Trajectories

The trajectory engine allows a vehicle to be searched using its license plate.

Conceptually:

```python
trajectory = engine.get_trajectory(
    plate_number="XX00XX0000",
    fuzzy=True
)
```

The engine can reconstruct the chronological sequence of cameras visited by that vehicle.

It also supports optional start/end timestamps and fuzzy plate matching to handle OCR imperfections.

---

#  Example Workflow

```text
Camera 1
   │
   │ Plate: XX00XX0000
   ▼
ANPR Detection
   │
   ▼
Camera 2
   │
   │ Plate: XX00XX0000
   ▼
ANPR Detection
   │
   ▼
Camera 3
   │
   │ Plate: XX00XX0000
   ▼
ANPR Detection
   │
   ▼
Trajectory Engine
   │
   ▼
GPS + Timestamp Association
   │
   ▼
Interactive Map
```

Result:

```text
                 CAM 02
                   ●
                  / \
                 /   \
                /     \
          CAM 01 ●──────● CAM 03

             Vehicle Route
```

---

#  Use Cases

This architecture can be extended for:

*  Urban traffic monitoring
*  Smart parking systems
*  Traffic flow analysis
*  Vehicle journey analysis
*  Smart-city infrastructure
*  Traffic-event investigation
*  Road utilization analytics
*  Multi-camera movement visualization

---

#  Future Improvements

Potential improvements include:

* [ ] Stronger cross-camera vehicle re-identification
* [ ] Dedicated vehicle appearance embeddings
* [ ] Improved low-light plate recognition
* [ ] Better handling of motorcycle plates
* [ ] Better handling of angled/bent plates
* [ ] GPU-accelerated inference
* [ ] RTSP/ONVIF camera support
* [ ] Database-backed event storage
* [ ] Real-time WebSocket dashboard
* [ ] Traffic density analytics
* [ ] Automatic congestion detection
* [ ] Vehicle speed estimation
* [ ] Advanced anomaly detection
* [ ] Containerized deployment
* [ ] Cloud-based multi-camera processing

---

#  Limitations

The current system is a prototype/research-oriented implementation.

Performance can be affected by:

* Camera resolution
* Motion blur
* Lighting conditions
* Plate angle
* Occlusion
* OCR errors
* Network latency
* Camera placement
* Number of simultaneous streams

The multi-camera implementation itself contains TODOs around live-feed speed, per-camera frame-rate control, and difficult plate geometries such as bent or multi-line motorcycle plates.

---

#  Privacy & Responsible Use

License plate information can constitute sensitive vehicle-related data depending on how and where the system is deployed.

When deploying this project in real environments:

* Use cameras only where legally permitted.
* Restrict access to collected ANPR data.
* Avoid unnecessary long-term storage.
* Protect stored plate information.
* Follow applicable privacy and data-protection requirements.
* Use the system for legitimate traffic-management or research purposes.

---

#  License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

#  Author

# [!Yuwin2008](https://github.com/Yuwin2008)
# [!Waqar Ahamad Khan]()
# [!Sairaj Dilip Vernekar](https://github.com/sdv-chess)
# [!Diya Prince](https://github.com/diya25prince-tech)

Project:

https://github.com/Yuwin2008/Urban-Tracking-ANPR-System

---

#  Project Vision

> **From detecting vehicles to understanding how they move through a city.**

Urban-Tracking-ANPR-System aims to provide a foundation for building intelligent, scalable, multi-camera traffic analytics systems where individual ANPR detections become part of a larger picture of urban vehicle movement.

If you find the project useful, consider giving it a star on GitHub.
