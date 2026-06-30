"""
Helper utilities for the Hand Gesture AI Chatbot
"""
import os
import json
import pickle
import numpy as np
from typing import Any, Dict, List, Optional
import cv2
from datetime import datetime

def ensure_directory_exists(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_json_file(data: Dict, filepath: str):
    """Save data to JSON file"""
    try:
        ensure_directory_exists(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def load_pickle_file(filepath: str) -> Optional[Any]:
    """Load pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_pickle_file(data: Any, filepath: str):
    """Save data to pickle file"""
    try:
        ensure_directory_exists(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)
    
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create black canvas and center the image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def get_file_size(filepath: str) -> str:
    """Get human-readable file size"""
    if not os.path.exists(filepath):
        return "0 B"
    
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def validate_landmarks(landmarks: List[float]) -> bool:
    """Validate hand landmarks data.
    MediaPipe normalized coords: x,y in [0,1]; z is depth (can be any float).
    """
    if len(landmarks) != 63:  # 21 landmarks * 3 coordinates
        return False
    
    # Check for NaN or infinite values (no range check on values — z can be outside [0,1])
    for value in landmarks:
        if np.isnan(value) or np.isinf(value):
            return False
    
    return True

def normalize_confidence(confidence: float) -> float:
    """Normalize confidence value to [0, 1]"""
    return max(0.0, min(1.0, confidence))

def create_backup_file(filepath: str) -> str:
    """Create backup of file"""
    if not os.path.exists(filepath):
        return ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(filepath, backup_path)
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {e}")
        return ""

def cleanup_old_backups(directory: str, max_backups: int = 5):
    """Clean up old backup files"""
    try:
        backup_files = []
        for file in os.listdir(directory):
            if file.endswith('.backup_'):
                filepath = os.path.join(directory, file)
                backup_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda x: x[1])
        
        # Remove oldest backups if too many
        if len(backup_files) > max_backups:
            for filepath, _ in backup_files[:-max_backups]:
                os.remove(filepath)
                print(f"Removed old backup: {filepath}")
    
    except Exception as e:
        print(f"Error cleaning up backups: {e}")

def log_activity(activity: str, log_file: str = "activity.log"):
    """Log activity to file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {activity}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    
    except Exception as e:
        print(f"Error logging activity: {e}")

def get_system_info() -> Dict:
    """Get system information"""
    import platform
    import psutil
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available
    }

# Test helper functions
def test_helpers():
    """Test helper functions"""
    print("=== Testing Helper Functions ===")
    
    # Test directory creation
    test_dir = "test_directory"
    ensure_directory_exists(test_dir)
    print(f"Directory created: {os.path.exists(test_dir)}")
    
    # Test JSON operations
    test_data = {"test": "data", "number": 42}
    json_file = os.path.join(test_dir, "test.json")
    save_json_file(test_data, json_file)
    loaded_data = load_json_file(json_file)
    print(f"JSON test: {loaded_data == test_data}")
    
    # Test pickle operations
    pickle_file = os.path.join(test_dir, "test.pkl")
    save_pickle_file(test_data, pickle_file)
    loaded_pickle = load_pickle_file(pickle_file)
    print(f"Pickle test: {loaded_pickle == test_data}")
    
    # Test landmark validation
    valid_landmarks = [0.5] * 63
    invalid_landmarks = [0.5] * 62 + [2.0]  # Invalid coordinate
    print(f"Valid landmarks test: {validate_landmarks(valid_landmarks)}")
    print(f"Invalid landmarks test: {validate_landmarks(invalid_landmarks)}")
    
    # Test confidence normalization
    print(f"Confidence normalization: {normalize_confidence(1.5)}")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System: {sys_info['platform']} {sys_info['architecture']}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("Test completed!")

if __name__ == "__main__":
    test_helpers()
