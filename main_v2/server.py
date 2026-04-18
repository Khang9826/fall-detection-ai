# server.py  —  AI Fall Detection System (main entry point)
import os
import glob
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from camera import Camera
from fall_detection_v2 import AdvancedFallDetector
from config_backend import YOLO_MODEL_PATH, CAMERA_INDEX

app = Flask(__name__)
CORS(app)

# ── Upload folder ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv",
                ".jpg", ".jpeg", ".png", ".bmp"}

# ── Global state ──────────────────────────────────────────────────────────────
camera   = Camera(src=CAMERA_INDEX)
detector = AdvancedFallDetector(yolo_model_path=YOLO_MODEL_PATH)

latest_frame   = None
fps            = 0.0
fall           = False
fall_count     = 0
confidence     = 0.0
stability      = 0.0
current_source = "webcam"

lock = threading.Lock()

# ── Camera / inference thread ─────────────────────────────────────────────────
def camera_loop():
    global latest_frame, fps, fall, fall_count, confidence, stability

    prev_time = time.time()
    camera.start()

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, num_new_falls, num_persons = detector.process_frame(frame)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else fps
        prev_time = now

        fall_count = detector.total_fall_events

        # FIX: num_new_falls is only > 0 on the single transition frame.
        # Also hold the flag for every frame the FSM stays in FALL or LYING,
        # so the 500 ms frontend poll always catches it.
        fall_active = num_new_falls > 0
        if not fall_active and detector.person_tracker.tracks:
            for track in detector.person_tracker.tracks.values():
                if track.fsm.current_state.value in ("FALL", "LYING"):
                    fall_active = True
                    break

        # Derive stability from the first available FSM
        if detector.person_tracker.tracks:
            first_track = next(iter(detector.person_tracker.tracks.values()))
            stability = min(1.0, first_track.fsm.state_frame_count / 30.0)

        fall = fall_active

        with lock:
            latest_frame = annotated


# ── Stream generator ──────────────────────────────────────────────────────────
def generate_stream():
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({
        "fps":        round(fps, 1),
        "fall":       fall,
        "fall_count": fall_count,
        "confidence": round(confidence, 3),
        "stability":  round(stability, 3),
        "source":     current_source,
    })


@app.route("/start", methods=["POST"])
def start():
    return jsonify(status="running")


@app.route("/stop", methods=["POST"])
def stop():
    return jsonify(status="stopped")


# ── Upload & source switching ─────────────────────────────────────────────────
@app.route("/upload_media", methods=["POST"])
def upload_media():
    if "file" not in request.files:
        return jsonify(error="No file in request"), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify(error=f"Unsupported type '{ext}'"), 415

    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_FOLDER, filename)
    f.save(dest)
    print(f"[server] Uploaded: {dest}")

    _switch_source(dest, filename)
    return jsonify(ok=True, filename=filename)


@app.route("/set_source", methods=["POST"])
def set_source():
    data   = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "webcam").strip()

    if source == "webcam":
        _switch_source(CAMERA_INDEX, "webcam")
        return jsonify(ok=True, source="webcam")

    path = os.path.join(UPLOAD_FOLDER, secure_filename(source))
    if not os.path.isfile(path):
        return jsonify(error=f"File not found: {source}"), 404

    _switch_source(path, source)
    return jsonify(ok=True, source=source)


@app.route("/media_list")
def media_list():
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(UPLOAD_FOLDER, f"*{ext}")))
    names = sorted(os.path.basename(f) for f in files)
    return jsonify(files=names)


def _switch_source(src, label: str):
    global current_source
    camera.switch_source(src)
    detector.reset()
    with lock:
        global latest_frame
        latest_frame = None
    current_source = label
    print(f"[server] Source → {label}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AI FALL DETECTION SYSTEM")
    print("  http://localhost:5000")
    print("=" * 65 + "\n")

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)