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
CAMERA_INDEX = 0         # 0 for default webcam, 1 for external, or path to video file
FRAME_WIDTH = 640        # Resize input for faster processing (optional)
FRAME_HEIGHT = 480
FPS_LIMIT = 30           # Target FPS for the loop

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================
# YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training', 'yolo12s.pt')
YOLO_MODEL_PATH = r"G:\VLU\NĂM 2\HỌC KÌ 2\Machine Learning\project\experiments\checkpoints\v3\fall_detection_v2\weights\best.pt"  # Path to your YOLO model weights
YOLO_IMGSZ = 416                     # YOLO input image size
USE_GPU = "auto"                     # "auto", "cuda", or "cpu"
MEDIAPIPE_MODEL_COMPLEXITY = 0       # 0=fastest, 1=balanced, 2=most accurate
DEBUG_VISUALIZATION = False          # Show debug feature info on frame