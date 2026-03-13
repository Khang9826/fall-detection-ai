import os

# ============================================================================
# FLASK SERVER CONFIGURATION
# ============================================================================
SERVER_HOST = "0.0.0.0"  # '0.0.0.0' allows access from other devices on the network
SERVER_PORT = 5000
DEBUG = False            # Set to False for production/demo

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
CAMERA_INDEX = 0       # 0 for default webcam, 1 for external, or path to video file
FRAME_WIDTH = 640        # Resize input for faster processing (optional)
FRAME_HEIGHT = 480
FPS_LIMIT = 30           # Target FPS for the loop

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================
# We use a relative path so it works on any machine. 
# Ensure yolov8m.pt is in the project root or it will be downloaded.
YOLO_MODEL_PATH = "yolov8m.pt"
TRACKER_TYPE = "botsort"  # Tracker type: "botsort", "bytetrack", etc.