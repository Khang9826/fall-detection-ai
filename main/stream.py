import cv2
from camera import Camera
from fall_detection_v5 import TrackingFallDetector

camera = Camera(0)
ai = TrackingFallDetector(r"G:\VLU\ultralytics\yolov8\yolov8m.pt")

def generate_frames():
    camera.start()
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame = ai.process(frame)
        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
