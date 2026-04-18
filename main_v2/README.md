# 🛡️ AI Fall Detection System

Real-time fall detection for healthcare and smart surveillance, powered by **YOLOv8** person detection, **MediaPipe Pose** estimation, and a **Finite State Machine (FSM)** for temporally-aware, low-false-positive alerting.

---

## ✨ Features

- **Real-time detection** via webcam or uploaded video/image file
- **Finite State Machine** — STANDING → TRANSITION → FALL → LYING, with multi-frame confirmation to suppress false positives
- **Persistent fall alert** — banner stays active until the person stands back up
- **Live web dashboard** — stream, FPS counter, confidence & stability bars, alert log
- **Hot-swap sources** — switch between webcam, video files, and images without restarting
- **Upload & switch** — drag-and-drop media upload from the browser

---

## 🗂️ Project Structure

```
project/
├── server.py               # Main entry point (recommended)
├── app.py                  # Alternative Flask entry point (with FallAlertManager)
├── fall_detection_v2.py    # Core detection engine (YOLO + MediaPipe + FSM)
├── camera.py               # Thread-safe camera abstraction
├── fall_alert_manager.py   # Per-person alert state tracker
├── config_backend.py       # Central configuration
├── stream.py               # Minimal standalone stream (dev/test only)
├── requirements.txt        # Python dependencies
├── uploads/                # Auto-created — stores uploaded media files
└── templates/
    └── index(second).html  # Web dashboard UI
```

---

## ⚙️ Requirements

- Python **3.9+**
- A webcam **or** a video file to analyse

### Python packages

```
opencv-python == 4.12.0.88
numpy         == 1.26.4
mediapipe     == 0.10.21
flask         == 3.1.2
flask-cors
werkzeug
ultralytics   # YOLOv8
```

Install everything at once:

```bash
pip install -r requirements.txt
pip install flask-cors ultralytics
```

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <repo-url>
cd fall-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install flask-cors ultralytics
```

### 3. Get the YOLOv8 model

The system uses **YOLOv8m** for person detection. Place `yolov8m.pt` in the project root, or let Ultralytics download it automatically on first run:

```bash
# Optional: pre-download to avoid delay on first start
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 4. Run the server

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

---

## 🖥️ Web Dashboard

| Control | Description |
|---------|-------------|
| **▶ Start Camera** | Activates the live video stream and detection |
| **■ Stop** | Pauses the stream |
| **Source dropdown** | Switch between webcam and previously uploaded files |
| **⟳ Refresh List** | Reloads the list of uploaded files |
| **📁 Choose File → ⬆ Upload & Switch** | Upload a video or image and immediately analyse it |

### Status indicators

| Indicator | Meaning |
|-----------|---------|
| 🟢 **SAFE** | No fall detected |
| 🔴 **⚠ FALL DETECTED** | Fall alert active — clears when person stands up |
| **FPS** | Frames processed per second |
| **Fall Events** | Cumulative confirmed fall count |
| **Confidence bar** | YOLO detection confidence |
| **Stability bar** | How long current FSM state has been held |

---

## 🧠 How It Works

### Detection pipeline

```
Webcam / Video File
        │
        ▼
  YOLOv8 Person Detection
  (class 0 only, conf ≥ 0.5)
        │
        ▼
  MediaPipe Pose Estimation
  (per-person crop)
        │
        ▼
  Feature Extraction
  ┌─ Torso angle from vertical
  ├─ Body height/width ratio
  ├─ Hip–ankle vertical distance
  └─ Knee bend angle
        │
        ▼
  Finite State Machine (per person)
  STANDING → TRANSITION → FALL → LYING
        │
        ▼
  Alert Manager
  (persistent flag, cleared on STANDING)
```

### FSM states

| State | Meaning |
|-------|---------|
| **STANDING** | Upright — torso angle < 30°, body ratio > 1.8 |
| **TRANSITION** | Bending / sitting — intermediate angles |
| **FALL** | Rapid posture change detected — triggers alert |
| **LYING** | Horizontal posture — confirms fall |

A fall alert is raised only when the FSM transitions `STANDING/TRANSITION → FALL` with sufficient confidence. The alert stays active until the FSM returns to **STANDING**, preventing false clears from noisy frames.

---

## ⚙️ Configuration

All tunable parameters live in `config_backend.py` and the constants at the top of `fall_detection_v2.py`.

### `config_backend.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Default webcam index |
| `YOLO_MODEL_PATH` | `"yolov8m.pt"` | Path to YOLO weights |
| `SERVER_HOST` | `"0.0.0.0"` | Bind address (use `127.0.0.1` for local only) |
| `SERVER_PORT` | `5000` | HTTP port |

### `fall_detection_v2.py` — key thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum YOLO detection confidence |
| `MIN_FRAMES_FOR_STATE_CHANGE` | `3` | Frames needed to confirm an FSM transition |
| `FALL_COOLDOWN_FRAMES` | `30` | Frames before another fall can be registered |
| `RAPID_ANGLE_CHANGE_THRESHOLD` | `15.0°` | Per-frame torso angle change for fall detection |
| `STANDING_TORSO_ANGLE_MAX` | `30.0°` | Max torso angle to classify as STANDING |
| `LYING_TORSO_ANGLE_MIN` | `60.0°` | Min torso angle to classify as LYING |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/video_feed` | MJPEG video stream |
| `GET` | `/status` | JSON status (fps, fall flag, fall count, source) |
| `POST` | `/start` | Start/resume detection |
| `POST` | `/stop` | Stop detection |
| `POST` | `/upload_media` | Upload a video or image file |
| `POST` | `/set_source` | Switch active source by filename or `"webcam"` |
| `GET` | `/media_list` | List uploaded files |

### `/status` response example

```json
{
  "fps": 24.3,
  "fall": false,
  "fall_count": 1,
  "confidence": 0.872,
  "stability": 0.933,
  "source": "webcam"
}
```

---

## 🗃️ Supported Media Formats

| Type | Extensions |
|------|------------|
| Video | `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv` |
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp` |

---

## 🐛 Troubleshooting

**Camera not opening**
- Check that no other application is using the webcam.
- Try changing `CAMERA_INDEX` to `1` in `config_backend.py`.

**`yolov8m.pt` not found**
- Run `python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"` to download it, or place the file manually in the project root.

**Fall banner never appears**
- Ensure `fall_detection_v2.py` includes the `fall_alert_active` flag and `any_fall_active` property (see latest version).
- Check that `server.py` reads `detector.any_fall_active`, not raw FSM state values.

**False fall alerts (no fall occurred)**
- Increase `MIN_FRAMES_FOR_STATE_CHANGE` (e.g. `5`) to require more consecutive frames before confirming a transition.
- Increase `YOLO_CONFIDENCE_THRESHOLD` (e.g. `0.65`) to reject low-quality detections.
- The `LYING` state alone does **not** trigger an alert — only an actual `FALL` transition does.

---

## 📄 License

For academic and research use. See individual file headers for author information.
