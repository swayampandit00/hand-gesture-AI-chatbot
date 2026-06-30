"""
Hand Tracking Module — MediaPipe 0.10.x Tasks API
Part 1: Detect hand and extract landmarks

Uses mp.tasks.vision.HandLandmarker (new API for mediapipe >= 0.10).
Falls back to OpenCV skin-colour contour detection if mediapipe fails,
the .task model file is not found, or Python >= 3.13 (unsupported by MediaPipe).

MediaPipe officially supports Python 3.10–3.12 only.
On Python 3.13+ the C-extension hangs — we skip it entirely.
"""
import cv2
import numpy as np
import os
import sys
import threading

from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE, MAX_NUM_HANDS
)

# ── Python version guard ──────────────────────────────────────────────────────
# MediaPipe's C-extensions hang indefinitely on Python 3.13+.
# Skip ALL mediapipe attempts on unsupported versions.
_PYTHON_SUPPORTED = sys.version_info < (3, 13)
if not _PYTHON_SUPPORTED:
    print(
        f"[HandTracker] Python {sys.version_info.major}.{sys.version_info.minor} "
        "detected. MediaPipe requires Python 3.10–3.12. "
        "Using OpenCV fallback mode."
    )

# Path to the hand landmarker task model (mediapipe 0.10+ format)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "hand_landmarker.task"
)

# Module-level flag — set during lazy init
MEDIAPIPE_AVAILABLE = None  # None = not yet tested


def _check_mediapipe():
    """Quick check if mediapipe package is installed (does NOT load model).
    Always returns False on unsupported Python versions to avoid hangs.
    """
    if not _PYTHON_SUPPORTED:
        return False
    try:
        import importlib.util
        spec = importlib.util.find_spec("mediapipe")
        return spec is not None
    except Exception:
        return False


MEDIAPIPE_INSTALLED = _check_mediapipe()


class HandTracker:
    def __init__(self):
        self.cap = None
        self._landmarker = None        # mp.tasks.vision.HandLandmarker instance
        self._mp_image_cls = None      # mediapipe.Image class
        self._running_mode = None
        self._mediapipe_ready = False  # True once model is loaded
        self._last_result = None       # Cache for async/video-stream mode

    # ─────────────────────────────────────────────
    # MEDIAPIPE LAZY INITIALISER (Tasks API)
    # ─────────────────────────────────────────────

    def _init_mediapipe(self):
        """Load HandLandmarker from .task file on first use.
        Immediately returns False on Python 3.13+ (avoids C-ext hangs).
        """
        global MEDIAPIPE_AVAILABLE
        if self._mediapipe_ready:
            return True
        if not _PYTHON_SUPPORTED or not MEDIAPIPE_INSTALLED:
            MEDIAPIPE_AVAILABLE = False
            return False
        if not os.path.isfile(_MODEL_PATH):
            print(
                f"MediaPipe init failed: model file not found at '{_MODEL_PATH}'. "
                "Falling back to OpenCV detection."
            )
            MEDIAPIPE_AVAILABLE = False
            return False
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            self._mp_image_cls = mp.Image
            self._running_mode = RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=RunningMode.IMAGE,           # sync, frame-by-frame
                num_hands=MAX_NUM_HANDS,
                min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
            )

            self._landmarker = HandLandmarker.create_from_options(options)
            self._mediapipe_ready = True
            MEDIAPIPE_AVAILABLE = True
            print("MediaPipe HandLandmarker initialized successfully.")
            return True

        except Exception as e:
            print(f"MediaPipe init failed: {e}. Falling back to OpenCV detection.")
            MEDIAPIPE_AVAILABLE = False
            self._mediapipe_ready = False
            return False

    # ─────────────────────────────────────────────
    # CAMERA
    # ─────────────────────────────────────────────

    def initialize_camera(self, camera_index: int = None):
        """Initialize webcam. Also lazily loads MediaPipe in background."""
        if camera_index is None:
            camera_index = CAMERA_INDEX

        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        self.cap = None

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(camera_index, backend)
            except Exception:
                self.cap = cv2.VideoCapture(camera_index)

            if self.cap is not None and self.cap.isOpened():
                break
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        # Try next camera index if the first one failed
        if self.cap is None or not self.cap.isOpened():
            for idx in range(camera_index + 1, camera_index + 3):
                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(idx, backend)
                    except Exception:
                        self.cap = cv2.VideoCapture(idx)

                    if self.cap is not None and self.cap.isOpened():
                        camera_index = idx
                        break
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                if self.cap is not None and self.cap.isOpened():
                    break

        if not self.cap or not self.cap.isOpened():
            print(f"Camera init failed: could not open camera index {camera_index}")
            return False

        print(f"Camera initialized on index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Load MediaPipe in background so camera feed starts immediately
        threading.Thread(target=self._init_mediapipe, daemon=True).start()
        return True

    def get_frame(self):
        """Get a flipped (mirrored) frame from the camera."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.flip(frame, 1)
        return None

    # ─────────────────────────────────────────────
    # HAND DETECTION
    # ─────────────────────────────────────────────

    def detect_hands(self, frame):
        """
        Detect hands and return (landmarks_list, annotated_frame).

        landmarks_list: list of flat [63-float] arrays (one per detected hand).
        annotated_frame: frame with visual overlays.
        """
        if frame is None:
            return [], frame

        if self._mediapipe_ready:
            return self._detect_hands_mediapipe(frame)
        else:
            return self._detect_hands_opencv(frame)

    def _detect_hands_mediapipe(self, frame):
        """Detect hands using MediaPipe Tasks HandLandmarker."""
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_img)
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return self._detect_hands_opencv(frame)

        hand_landmarks_list = []
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # Flatten to [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] = 63 floats
                landmarks = []
                for lm in hand_lms:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hand_landmarks_list.append(landmarks)

                # Draw skeleton overlay manually
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                CONNECTIONS = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for a, b in CONNECTIONS:
                    cv2.line(annotated_frame, points[a], points[b], (0, 255, 0), 2)
                for pt in points:
                    cv2.circle(annotated_frame, pt, 4, (0, 0, 255), -1)

        return hand_landmarks_list, annotated_frame

    def _detect_hands_opencv(self, frame):
        """Fallback: simple skin-colour contour detection (no MediaPipe)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hand_landmarks_list = []
        annotated_frame = frame.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                cv2.drawContours(annotated_frame, [largest], 0, (0, 255, 0), 2)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    landmarks = []
                    for _ in range(21):
                        landmarks.extend([cx / frame.shape[1], cy / frame.shape[0], 0.0])
                    hand_landmarks_list.append(landmarks)
                    cv2.circle(annotated_frame, (cx, cy), 10, (255, 0, 0), -1)

        # Show loading overlay if mediapipe is still initialising
        if not self._mediapipe_ready and MEDIAPIPE_INSTALLED:
            cv2.putText(
                annotated_frame,
                "Loading MediaPipe...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
            )
        elif not MEDIAPIPE_INSTALLED:
            cv2.putText(
                annotated_frame,
                "OpenCV mode (no MediaPipe)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
            )

        return hand_landmarks_list, annotated_frame

    # ─────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────

    def extract_landmark_features(self, hand_landmarks):
        """Extract normalised features from a flat list of 63 landmark floats.

        Args:
            hand_landmarks: List of 63 floats (21 landmarks × [x, y, z])
        Returns:
            1-D numpy array of shape (63,), or None on error.
        """
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return None

        try:
            landmarks = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)

            # Translate wrist to origin
            wrist = landmarks[0].copy()
            normalized = landmarks - wrist

            # Scale for size invariance
            scale = np.max(np.abs(normalized)) + 1e-8
            normalized = normalized / scale

            return normalized.flatten()   # shape (63,)
        except Exception as e:
            print(f"extract_landmark_features error: {e}")
            return None

    # ─────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────

    def release(self):
        """Release camera and MediaPipe resources."""
        if self.cap:
            self.cap.release()
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# TEST UTILITY
# ─────────────────────────────────────────────

def test_hand_tracker():
    """Test the hand tracker interactively."""
    tracker = HandTracker()

    if not tracker.initialize_camera():
        print("Failed to initialize camera")
        return

    print("Hand Tracker Test — Press 'q' to quit")
    print("(MediaPipe may take a moment to load on first run)")

    while True:
        frame = tracker.get_frame()
        if frame is None:
            break

        hand_landmarks, annotated_frame = tracker.detect_hands(frame)

        if hand_landmarks:
            features = tracker.extract_landmark_features(hand_landmarks[0])
            label = f"Hand detected | features: {features.shape if features is not None else 'None'}"
        else:
            label = "No hand detected"

        cv2.putText(annotated_frame, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.release()


if __name__ == "__main__":
    test_hand_tracker()
