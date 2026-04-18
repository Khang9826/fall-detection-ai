import os
import glob
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import threading
import atexit
import time
from collections import deque
from pathlib import Path

from fall_detection_v2 import AdvancedFallDetector, YOLO_MODEL_PATH
from fall_alert_manager import FallAlertManager

app = Flask(__name__)
CORS(app)

# ── Upload folder ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================================
# GLOBAL STATE
# ============================================================================

detector = None
is_running = False
lock = threading.Lock()

camera_manager = None

current_fps = 0
total_fall_count = 0
avg_confidence = 0.85
avg_stability = 0.92
is_fall_detected = False
latest_alert_events = []

current_source = "webcam"

alert_manager = FallAlertManager(
    fall_confirm_frames=3,
    risk_confirm_frames=5,
    resolve_frames=5,
    cooldown_sec=10.0,
    min_confidence=0.6
)

fps_deque = deque(maxlen=30)
last_frame_time = time.time()

# ============================================================================
# FALL DETECTION INTEGRATION
# ============================================================================

def initialize_detector():
    global detector
    if detector is None:
        detector = AdvancedFallDetector(yolo_model_path=YOLO_MODEL_PATH)
        print("[Flask] Fall Detection Model Ready!")
    return detector


class CameraManager:
    """
    Dedicated capture thread that keeps only the latest frame.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self._cap = None
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None
        self._last_read_ok = True
        self._reconnect_backoff = 1.0
        self._last_reconnect_attempt = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._release()

    def switch_source(self, new_source):
        with self._lock:
            self._release()
            self.camera_index = new_source
            self._latest_frame = None
            self._last_reconnect_attempt = 0.0

    def _open(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)

    def _release(self):
        if self._cap:
            self._cap.release()
        self._cap = None

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                now = time.time()
                if now - self._last_reconnect_attempt >= self._reconnect_backoff:
                    self._last_reconnect_attempt = now
                    self._release()
                    self._open()
                time.sleep(0.05)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._last_read_ok = False
                if isinstance(self.camera_index, str) and os.path.isfile(str(self.camera_index)):
                    with self._lock:
                        if self._cap:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                else:
                    self._release()
                    time.sleep(0.05)
                continue

            with self._lock:
                self._latest_frame = frame
            self._last_read_ok = True

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


def start_camera():
    global is_running, detector, camera_manager
    with lock:
        if detector is None:
            detector = initialize_detector()
        if camera_manager is None:
            camera_manager = CameraManager(camera_index=0)
        if not camera_manager._running:
            camera_manager.start()
        is_running = True
        print("[Flask] Camera started successfully")


def stop_camera():
    global is_running, camera_manager
    with lock:
        if is_running:
            if camera_manager:
                camera_manager.stop()
            is_running = False
            print("[Flask] Camera stopped")


def get_frame():
    global detector, current_fps, total_fall_count
    global avg_confidence, avg_stability, is_fall_detected, latest_alert_events
    global last_frame_time, fps_deque

    with lock:
        _detector = detector
        _camera   = camera_manager

    if _camera is None or _detector is None:
        return None

    current_time = time.time()
    time_diff = current_time - last_frame_time
    if time_diff > 0:
        fps_deque.append(1.0 / time_diff)
        _fps = int(np.mean(fps_deque))
    else:
        _fps = current_fps
    last_frame_time = current_time

    frame = _camera.get_latest_frame()
    if frame is None:
        return None

    try:
        annotated_frame, num_new_falls, num_persons = _detector.process_frame(frame)

        _total_falls = _detector.total_fall_events
        _conf        = avg_confidence
        _stab        = avg_stability
        events       = []
        now_ts       = time.time()

        # Detect fall state: FALL or NO_FALL
        _fall_detected = num_new_falls > 0
        if not _fall_detected and _detector.person_tracker.tracks:
            for track in _detector.person_tracker.tracks.values():
                if (track.fsm.current_state.value == "FALL" or
                        track.fsm.fall_alert_counter > 0):
                    _fall_detected = True
                    break

        # Use FallAlertManager for multi-frame confirmation + cooldown
        if _detector.person_tracker.tracks:
            for person_id, track in _detector.person_tracker.tracks.items():
                state_val = track.fsm.current_state.value
                conf = track.fsm.fall_confidence if track.fsm.fall_confidence > 0 else _conf
                bbox = track.bbox

                # Two-state mapping: FALL or NORMAL
                if state_val == "FALL" or track.fsm.fall_alert_counter > 0:
                    alert_state = "FALL"
                else:
                    alert_state = "NORMAL"

                person_events = alert_manager.update(
                    person_id=person_id,
                    state=alert_state,
                    confidence=conf,
                    bbox=bbox,
                    timestamp=now_ts
                )
                events.extend(person_events)

            first_track = next(iter(_detector.person_tracker.tracks.values()))
            _stab = min(1.0, first_track.fsm.state_frame_count / 30.0)

        current_fps         = _fps
        total_fall_count    = _total_falls
        is_fall_detected    = _fall_detected
        avg_confidence      = _conf
        avg_stability       = _stab
        latest_alert_events = events

        _, buffer = cv2.imencode('.jpg', annotated_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buffer.tobytes()

    except Exception as e:
        print(f"[Flask] Detection error: {e}")
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()


def camera_loop():
    while True:
        if not is_running:
            time.sleep(0.1)
            continue

        frame = get_frame()
        if frame is None:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "running":    is_running,
        "fps":        current_fps,
        "fall_count": total_fall_count,
        "confidence": avg_confidence,
        "stability":  avg_stability,
        "fall":       is_fall_detected,
        "alerts":     latest_alert_events,
        "source":     current_source,
    })


@app.route("/start", methods=["POST"])
def start():
    start_camera()
    return jsonify({"status": "started", "running": is_running})


@app.route("/stop", methods=["POST"])
def stop():
    stop_camera()
    return jsonify({"status": "stopped", "running": is_running})


@app.route("/video_feed")
def video_feed():
    return Response(
        camera_loop(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# UPLOAD / SOURCE SWITCHING ROUTES
# ============================================================================

@app.route("/upload_media", methods=["POST"])
def upload_media():
    if "file" not in request.files:
        return jsonify(error="No file in request"), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify(error=f"Unsupported type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}"), 415

    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_FOLDER, filename)
    f.save(dest)
    print(f"[app] Uploaded: {dest}")

    _switch_to_file(dest, filename)
    return jsonify(ok=True, filename=filename)


@app.route("/set_source", methods=["POST"])
def set_source():
    data   = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "webcam").strip()

    if source == "webcam":
        _switch_to_webcam()
        return jsonify(ok=True, source="webcam")

    path = os.path.join(UPLOAD_FOLDER, secure_filename(source))
    if not os.path.isfile(path):
        return jsonify(error=f"File not found: {source}"), 404

    _switch_to_file(path, source)
    return jsonify(ok=True, source=source)


@app.route("/media_list")
def media_list():
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(UPLOAD_FOLDER, f"*{ext}")))
    names = sorted(os.path.basename(f) for f in files)
    return jsonify(files=names)


def _switch_to_webcam():
    global is_running, camera_manager, current_source, detector
    with lock:
        if detector is None:
            detector = initialize_detector()
        if camera_manager is None:
            camera_manager = CameraManager(camera_index=0)
            camera_manager.start()
        else:
            camera_manager.switch_source(0)
            if not camera_manager._running:
                camera_manager.start()
        detector.reset()
        is_running = True
        current_source = "webcam"
    print("[app] Source switched to: webcam")


def _switch_to_file(path: str, label: str):
    global is_running, camera_manager, current_source, detector
    with lock:
        if detector is None:
            detector = initialize_detector()
        if camera_manager is None:
            camera_manager = CameraManager(camera_index=path)
            camera_manager.start()
        else:
            camera_manager.switch_source(path)
            if not camera_manager._running:
                camera_manager.start()
        detector.reset()
        is_running = True
        current_source = label
    print(f"[app] Source switched to: {label}")


@atexit.register
def _shutdown():
    stop_camera()
    if detector:
        detector.close()


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AI FALL DETECTION SYSTEM - WEB SERVER")
    print("=" * 80)
    print("Starting Flask server on http://0.0.0.0:5000")
    print("Open http://localhost:5000 in your browser")
    print("=" * 80 + "\n")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_camera()
        if detector:
            detector.close()