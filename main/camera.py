# import cv2
# import threading

# class Camera:
#     def __init__(self, src=0):
#         self.src = src
#         self.cap = None
#         self.frame = None
#         self.running = False
#         self.lock = threading.Lock()

#     def start(self):
#         if self.running:
#             return
#         self.cap = cv2.VideoCapture(self.src)
#         self.running = True
#         threading.Thread(target=self.update, daemon=True).start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()





# camera.py
import cv2
import threading

class Camera:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None