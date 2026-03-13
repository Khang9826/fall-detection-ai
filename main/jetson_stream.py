import cv2
import requests

SEVER = "http://192.168.1.191:5000/upload"

pipeline = (
    " nvarguscamerasrc ! "
    " video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    " nvvidconv ! video/x-raw, format=BGRx ! "
    " videoconvert ! video/x-raw, format=BGR ! appsink "
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
  print("Cannot open camera")
  exit()

while True:
  ret, frame = cap.read()

  if not ret:
      print("Camera error")
      continue

_, buffer = cv2.imencode(".jpg",frame)

try:
  r = request.post(
      SERVER,
      files={"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
      timeout=10
  )
  pritn("status:", r.status_code)
except Exception as e:
  print("Cannot connect server:", e)
