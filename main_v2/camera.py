# camera.py
import cv2
import threading
import time
import os

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}


def _is_image(src) -> bool:
    return os.path.splitext(str(src))[1].lower() in SUPPORTED_IMAGE_EXTS


def _is_video_file(src) -> bool:
    return os.path.splitext(str(src))[1].lower() in SUPPORTED_VIDEO_EXTS


class Camera:
    """
    Thread-safe camera supporting:
      - Webcam index (int)            e.g. Camera(0)
      - Video file (.mp4/.avi/…)      loops automatically
      - Static image (.jpg/.png/…)    repeats same frame at ~30 fps
    """

    def __init__(self, src=0):
        self.src    = src
        self.cap    = None
        self._static_frame = None
        self.frame  = None
        self.running = False
        self.lock   = threading.Lock()
        self._thread = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self._open_source(self.src)
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def switch_source(self, src):
        """Hot-swap to a new source without stopping the reader thread."""
        with self.lock:
            self._release_cap()
            self.src           = src
            self._static_frame = None
            self.frame         = None
            self._open_source(src)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        with self.lock:
            self._release_cap()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_source(self, src):
        """Open cap or load image. Call inside lock or before thread starts."""
        if _is_image(src):
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"Cannot read image: {src}")
            self._static_frame = img
            self.frame         = img.copy()
            self.cap           = None
        else:
            index = int(src) if str(src).isdigit() else src
            cap   = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {src}")
            self.cap           = cap
            self._static_frame = None

    def _release_cap(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def _update(self):
        while self.running:
            with self.lock:
                static = self._static_frame
                cap    = self.cap
                src    = self.src

            # Static image — hold frame at ~30 fps
            if static is not None:
                with self.lock:
                    self.frame = static.copy()
                time.sleep(0.033)
                continue

            if cap is None:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                # Loop video files; try to reconnect webcam
                if _is_video_file(src):
                    with self.lock:
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                else:
                    with self.lock:
                        self._release_cap()
                    time.sleep(0.5)
                continue

            with self.lock:
                self.frame = frame