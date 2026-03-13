# import cv2
# import time
# import threading
# from flask import Flask, render_template, Response, jsonify
# from fall_detection_v6 import FallDetector

# app = Flask(__name__)


# detector = FallDetector()
# cap = cv2.VideoCapture(0)

# def gen_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         frame = detector.process_frame(frame)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# # =====================
# # GLOBAL STATE
# # =====================
# camera = None
# detector = None

# latest_frame = None
# fps = 0.0
# fall = False
# fall_count = 0
# confidence = 0.0
# stability = 0.0

# lock = threading.Lock()


# # =====================
# # CAMERA LOOP
# # =====================
# def camera_loop():
#     global camera, detector
#     global latest_frame, fps, fall, fall_count, confidence, stability

#     prev_time = time.time()

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             continue

#         annotated, count = detector.process_frame(frame)

#         now = time.time()
#         fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
#         prev_time = now

#         # Heuristic metrics (safe for frontend)
#         fall = count > fall_count
#         fall_count = count
#         confidence = min(1.0, confidence * 0.9 + 0.1)
#         stability = min(1.0, stability * 0.95 + 0.05)

#         with lock:
#             latest_frame = annotated


# # =====================
# # ROUTES
# # =====================
# @app.route("/")
# def index():
#     return render_template("index.html")


# def generate_stream():
#     global latest_frame

#     while True:
#         with lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()

#         ret, buffer = cv2.imencode(".jpg", frame)
#         if not ret:
#             continue

#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" +
#             buffer.tobytes() +
#             b"\r\n"
#         )


# @app.route("/video_feed")
# def video_feed():
#     return Response(
#         gen_frames(),
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )


# @app.route("/status")
# def status():
#     return jsonify({
#         "fps": round(fps, 2),
#         "fall": fall,
#         "fall_count": fall_count,
#         "confidence": round(confidence, 2),
#         "stability": round(stability, 2)
#     })


# # =====================
# # MAIN
# # =====================
# if __name__ == "__main__":
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         raise RuntimeError("Cannot open camera")

#     detector = FallDetector()

#     t = threading.Thread(target=camera_loop, daemon=True)
#     t.start()

#     app.run(host="0.0.0.0", port=5000, debug=True)






# server.py
import numpy as np
import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
# from camera import Camera
from fall_detection_v5 import TrackingFallDetector
from config_backend import YOLO_MODEL_PATH, TRACKER_TYPE, CAMERA_INDEX

app = Flask(__name__)

# =====================
# GLOBAL STATE
# =====================
from camera import Camera
camera = Camera(src=CAMERA_INDEX)
detector = TrackingFallDetector(yolo_model_path=YOLO_MODEL_PATH,
                                tracker_type=TRACKER_TYPE)

latest_frame = None
fps = 30.0
fall = False
fall_count = 0
confidence = 0.0
stability = 0.0

lock = threading.Lock()

# =====================
# CAMERA LOOP
# =====================
def camera_loop():
    global latest_frame, fps, fall, fall_count, confidence, stability

    prev_time = time.time()
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        annotated, count = detector.process_frame(frame)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        # Metrics
        fall = count > fall_count
        fall_count = count
        confidence = min(1.0, confidence * 0.9 + 0.1)
        stability = min(1.0, stability * 0.95 + 0.05)

        with lock:
            latest_frame = annotated

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return render_template("index.html")

def generate_stream():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() +
               b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "fps": round(fps, 2),
        "fall": fall,
        "fall_count": fall_count,
        "confidence": round(confidence, 2),
        "stability": round(stability, 2)
    })

# @app.route("/start", methods=["POST"])
# def start():
#     camera.start()
#     threading.Thread(target=camera_loop, daemon=True).start()
#     return {"status": "started"}

# @app.route("/stop", methods=["POST"])
# def stop():
#     camera.stop()
#     return {"status": "stopped"}

@app.route("/upload", methods=["POST"])
def upload_frame():
    global latest_frame, fps, fall, fall_count
    print("Frame received")
    file = request.files["frame"]

    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    annotated, count = detector.process_frame(frame)

    with lock:
        latest_frame = annotated

    return {"status": "ok"}

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    # threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)