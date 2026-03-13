from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import atexit
import time
from collections import deque
from pathlib import Path

# Import fall detection components
# from fall_detection_v5 import FallDetector, TrackingFallDetector, YOLO_MODEL_PATH, TRACKER_TYPE
from fall_detection_v5 import TrackingFallDetector, YOLO_MODEL_PATH, TRACKER_TYPE
from fall_alert_manager import FallAlertManager

app = Flask(__name__)
CORS(app)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Camera and detection globals
detector = None
is_running = False
lock = threading.Lock()

# Threaded camera manager (single open, latest-frame buffer)
camera_manager = None

# Real-time metrics
current_fps = 0
total_fall_count = 0
avg_confidence = 0.85
avg_stability = 0.92
is_fall_detected = False
latest_alert_events = []

# Alert manager (per-person, multi-frame confirmation + cooldown)
alert_manager = FallAlertManager(
    fall_confirm_frames=3,
    risk_confirm_frames=5,
    resolve_frames=5,
    cooldown_sec=10.0,
    min_confidence=0.6
)

# FPS calculation
fps_deque = deque(maxlen=30)
last_frame_time = time.time()

# ============================================================================
# FALL DETECTION INTEGRATION
# ============================================================================

def initialize_detector():
    """Initialize the fall detection model once"""
    global detector
    if detector is None:
        detector = TrackingFallDetector(
            yolo_model_path=YOLO_MODEL_PATH,
            tracker_type=TRACKER_TYPE
        )
        print("[Flask] Fall Detection Model Ready!")
    return detector

class CameraManager:
    """
    Dedicated capture thread that keeps only the latest frame.
    The inference thread always reads the newest frame and drops stale frames.
    """
    def __init__(self, camera_index: int = 0):
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

    def _open(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)

    def _release(self):
        if self._cap:
            self._cap.release()
        self._cap = None

    def _capture_loop(self):
        while self._running:
            # Ensure camera is opened exactly once and auto-recover if needed
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
                # Temporary failure; release to trigger reconnect
                self._release()
                time.sleep(0.05)
                continue

            # Store only the newest frame (drop older frames)
            with self._lock:
                self._latest_frame = frame
            self._last_read_ok = True

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

def start_camera():
    """Start camera and initialize detector"""
    global is_running, detector, camera_manager
    with lock:
        if not is_running:
            detector = initialize_detector()
            if camera_manager is None:
                camera_manager = CameraManager(camera_index=0)
            camera_manager.start()
            is_running = True
            print("[Flask] Camera started successfully")

def stop_camera():
    """Stop camera and release resources"""
    global is_running, camera_manager
    with lock:
        if is_running:
            if camera_manager:
                camera_manager.stop()
            is_running = False
            print("[Flask] Camera stopped")

# def get_frame_with_detection():
def get_frame():
    """
    Capture frame, run fall detection, and return annotated frame
    Also updates global metrics
    """
    global detector, current_fps, total_fall_count
    global avg_confidence, avg_stability, is_fall_detected, total_fall_count, latest_alert_events
    global last_frame_time, fps_deque
    
    if camera_manager is None or detector is None:
        return None
    
    # Calculate FPS
    current_time = time.time()
    time_diff = current_time - last_frame_time
    if time_diff > 0:
        fps_deque.append(1.0 / time_diff)
        current_fps = int(np.mean(fps_deque))
    last_frame_time = current_time
    
    # Read latest frame from camera thread (drop stale frames)
    frame = camera_manager.get_latest_frame()
    if frame is None:
        return None
    
    # Run fall detection
    try:
        annotated_frame, num_new_falls = detector.process_frame(frame)
        
        # Update global metrics
        total_fall_count = detector.total_falls
        
        # Extract metrics from detector
        if detector.tracker.tracked_persons:
            # Get first tracked person's state
            first_person = next(iter(detector.tracker.tracked_persons.values()))
            
            # Update fall detection status
            is_fall_detected = (
                first_person.fall_detected or 
                first_person.fall_alert_counter > 0
            )
            
            # Update confidence from last features
            if first_person.last_valid_features:
                avg_confidence = first_person.last_valid_features.confidence
            
            # Calculate stability based on tracking age
            if first_person.age > 0:
                avg_stability = min(1.0, first_person.age / 30.0)
        else:
            is_fall_detected = False

        # Generate per-person alert events (multi-person support)
        events = []
        now_ts = time.time()
        for person in detector.tracker.tracked_persons.values():
            state_val = getattr(person.current_state, "value", str(person.current_state))
            # Map internal FSM states to alert states
            if state_val in ("FALL_CONFIRMED", "FALL"):
                alert_state = "FALL"
            elif state_val in ("FALLING", "UNSTABLE", "RISK", "TRANSITION"):
                alert_state = "RISK"
            elif state_val in ("LYING",):
                alert_state = "LYING"
            else:
                alert_state = "NORMAL"

            conf = 0.0
            if person.last_valid_features is not None:
                conf = float(person.last_valid_features.confidence)

            bbox = person.bbox
            events.extend(alert_manager.update(
                person_id=person.track_id,
                state=alert_state,
                confidence=conf,
                bbox=bbox,
                timestamp=now_ts
            ))

        latest_alert_events = events
        if any(ev["event_type"] == "fall_alert" for ev in events):
            is_fall_detected = True
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"[Flask] Detection error: {e}")
        # Return original frame on error
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

def camera_loop():
    """Generator for video streaming"""
    while True:
        if not is_running:
            time.sleep(0.1)
            continue
            
        # frame = get_frame_with_detection()
        frame = get_frame()
        if frame is None:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def _legacy_generate_frames():
    """Deprecated legacy capture loop (unused)."""
    return iter(())

    """

    while True:
        if not camera_on:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            break

        # ===== YOLO + MediaPipe xử lý ở đây =====
        # frame = process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    """

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route("/")
def index():
    """Serve main dashboard"""
    return render_template("index.html")

@app.route("/status")
def status():
    """Return current system status and metrics"""
    return jsonify({
        "running": is_running,
        "fps": current_fps,
        "fall_count": total_fall_count,
        "confidence": avg_confidence,
        "stability": avg_stability,
        "fall": is_fall_detected,
        "alerts": latest_alert_events
    })

@app.route("/start", methods=["POST"])
def start():
    """Start camera and detection"""
    start_camera()
    return jsonify({"status": "started", "running": is_running})
# def start_camera():
#     global camera_on
#     camera_on = True
#     return {"status": "started"}

@app.route("/stop", methods=["POST"])
def stop():
    """Stop camera and detection"""
    stop_camera()
    return jsonify({"status": "stopped", "running": is_running})
# def stop_camera():
#     global camera_on
#     camera_on = False
#     return {"status": "stopped"}

@app.route("/video_feed")
def video_feed():
    """Stream video with fall detection annotations"""
    return Response(
        camera_loop(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# Ensure resources are released on server shutdown
@atexit.register
def _shutdown():
    stop_camera()
    if detector:
        detector.close()

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AI FALL DETECTION SYSTEM - WEB SERVER")
    print("="*80)
    print("Starting Flask server on http://0.0.0.0:5000")
    print("Open http://localhost:5000 in your browser")
    print("="*80 + "\n")
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_camera()
        if detector:
            detector.close()
