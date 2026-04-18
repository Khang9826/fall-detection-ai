import cv2
from camera import Camera
from fall_detection_v2 import AdvancedFallDetector, YOLO_MODEL_PATH

camera = Camera(0)
ai = AdvancedFallDetector(yolo_model_path=YOLO_MODEL_PATH)

def generate_frames():
    camera.start()
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        annotated, _, _ = ai.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', annotated)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
