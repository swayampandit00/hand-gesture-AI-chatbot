"""
Configuration settings for Hand Gesture AI Chatbot
"""
import os

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe settings
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# Model settings
MODEL_PATH = "model/gesture_model.pkl"   # Primary model path (scikit-learn)
MODEL_PATH_H5 = "model/gesture_model.h5"  # TensorFlow model path (if TF available)
LABELS_PATH = "model/labels.pkl"
LANDMARK_COUNT = 21  # MediaPipe hand landmarks

# Gesture settings
GESTURE_CONFIDENCE_THRESHOLD = 0.6   # Lowered from 0.8 for better detection
GESTURE_HOLD_TIME = 1.0  # seconds
GESTURE_COOLDOWN = 3.0   # seconds between gesture commands sent to chatbot

# Voice settings
VOICE_RATE = 150
VOICE_VOLUME = 0.9

# Chatbot settings
CHATBOT_MODE = "rasa"  # "rasa" or "fallback"

# GUI settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FONT_SIZE = 12

# Dataset settings
DATASET_PATH = "dataset/samples"
SAMPLES_PER_GESTURE = 100
VALIDATION_SPLIT = 0.2

# Training settings
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Gesture to command mappings
DEFAULT_GESTURE_COMMANDS = {
    "thumbs_up": "hello",
    "peace": "time",
    "palm": "start_listening",
    "point": "select"
}
