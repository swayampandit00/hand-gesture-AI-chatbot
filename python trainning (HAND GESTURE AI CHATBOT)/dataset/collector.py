"""
Gesture Dataset Collector
Part 2: Collect gesture samples for training
"""
import cv2
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from vision.hand_tracker import HandTracker
from vision.landmark_extractor import LandmarkExtractor
from config.settings import DATASET_PATH, SAMPLES_PER_GESTURE, CAMERA_INDEX

class GestureDatasetCollector:
    def __init__(self):
        self.tracker = HandTracker()
        self.extractor = LandmarkExtractor()
        self.current_gesture = ""
        self.collected_samples = []
        self.is_collecting = False
        
        # Ensure dataset directory exists
        os.makedirs(DATASET_PATH, exist_ok=True)
        
    def get_available_gestures(self) -> List[str]:
        """Get list of already collected gestures"""
        if not os.path.exists(DATASET_PATH):
            return []
        
        gestures = []
        for item in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, item)):
                gestures.append(item)
        return sorted(gestures)
    
    def create_gesture_directory(self, gesture_name: str) -> str:
        """Create directory for new gesture"""
        gesture_dir = os.path.join(DATASET_PATH, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        return gesture_dir
    
    def save_sample(self, gesture_name: str, landmarks: List[float], frame: np.ndarray):
        """Save a single gesture sample"""
        gesture_dir = os.path.join(DATASET_PATH, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        landmark_file = os.path.join(gesture_dir, f"{timestamp}.json")
        image_file = os.path.join(gesture_dir, f"{timestamp}.jpg")
        
        # Save landmarks
        sample_data = {
            "gesture": gesture_name,
            "timestamp": timestamp,
            "landmarks": landmarks,
            "features": self.extractor.process_landmarks(landmarks).tolist()
        }
        
        with open(landmark_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Save image
        cv2.imwrite(image_file, frame)
        
        return landmark_file, image_file
    
    def save_gesture_metadata(self, gesture_name: str, command: str):
        """Save gesture metadata including command mapping"""
        gesture_dir = os.path.join(DATASET_PATH, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        metadata_file = os.path.join(gesture_dir, "metadata.json")
        metadata = {
            "gesture": gesture_name,
            "command": command,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata for gesture '{gesture_name}' with command: {command}")
    
    def load_gesture_metadata(self, gesture_name: str) -> Optional[Dict]:
        """Load gesture metadata including command mapping"""
        gesture_dir = os.path.join(DATASET_PATH, gesture_name)
        metadata_file = os.path.join(gesture_dir, "metadata.json")
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {gesture_name}: {e}")
            return None
    
    def load_all_gesture_mappings(self) -> Dict[str, str]:
        """Load all gesture-command mappings from dataset"""
        mappings = {}
        
        if not os.path.exists(DATASET_PATH):
            return mappings
        
        for gesture_name in os.listdir(DATASET_PATH):
            gesture_dir = os.path.join(DATASET_PATH, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue
            
            metadata = self.load_gesture_metadata(gesture_name)
            if metadata and "command" in metadata:
                mappings[gesture_name] = metadata["command"]
        
        return mappings
    
    def collect_gesture_samples(self, gesture_name: str, target_count: int = SAMPLES_PER_GESTURE):
        """Collect samples for a specific gesture"""
        print(f"Starting collection for gesture: {gesture_name}")
        print(f"Target: {target_count} samples")
        print("Press SPACE to collect sample, 'q' to quit, 'r' to reset")
        
        if not self.tracker.initialize_camera(camera_index=CAMERA_INDEX):
            print("Failed to initialize camera")
            return False

        window_name = 'Gesture Dataset Collector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        self.current_gesture = gesture_name
        self.collected_samples = []
        self.is_collecting = True
        
        collected_count = 0
        
        while self.is_collecting and collected_count < target_count:
            frame = self.tracker.get_frame()
            if frame is None:
                print("Warning: empty camera frame received")
                break
            
            hand_landmarks, annotated_frame = self.tracker.detect_hands(frame)
            
            # Display instructions
            cv2.putText(annotated_frame, f"Gesture: {gesture_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Collected: {collected_count}/{target_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "SPACE: Collect | Q: Quit | R: Reset", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if hand_landmarks:
                # Show hand detected
                cv2.putText(annotated_frame, "HAND DETECTED", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Extract features
                try:
                    features = self.extractor.process_landmarks(hand_landmarks[0])
                    cv2.putText(annotated_frame, f"Features: {len(features)}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Landmark processing warning: {e}")
            else:
                cv2.putText(annotated_frame, "HAND NOT DETECTED", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Place your hand in the camera frame", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow(window_name, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.is_collecting = False
                print("Collection stopped by user")
                break
            elif key == ord('r'):
                collected_count = 0
                self.collected_samples = []
                print("Collection reset")
            elif key == ord(' '):
                if hand_landmarks:
                    landmark_file, image_file = self.save_sample(
                        gesture_name, hand_landmarks[0], frame
                    )
                    self.collected_samples.append({
                        "landmark_file": landmark_file,
                        "image_file": image_file
                    })
                    collected_count += 1
                    print(f"Sample {collected_count}/{target_count} collected")
                    
                    annotated_frame[:] = 255
                    cv2.putText(annotated_frame, "CAPTURED!", 
                               (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow(window_name, annotated_frame)
                    cv2.waitKey(100)
                else:
                    print("No hand detected. Hold your hand in front of the camera and try again.")
        return collected_count > 0
    
    def load_dataset(self) -> Dict[str, List[np.ndarray]]:
        """Load all collected gesture data"""
        dataset = {}
        
        if not os.path.exists(DATASET_PATH):
            return dataset
        
        for gesture_name in os.listdir(DATASET_PATH):
            gesture_dir = os.path.join(DATASET_PATH, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue
            
            gesture_samples = []
            
            for file_name in os.listdir(gesture_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(gesture_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            sample_data = json.load(f)
                            if 'features' in sample_data:
                                gesture_samples.append(np.array(sample_data['features']))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            if gesture_samples:
                dataset[gesture_name] = gesture_samples
                print(f"Loaded {len(gesture_samples)} samples for gesture: {gesture_name}")
        
        return dataset
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the collected dataset"""
        dataset = self.load_dataset()
        stats = {
            "total_gestures": len(dataset),
            "total_samples": sum(len(samples) for samples in dataset.values()),
            "gesture_counts": {gesture: len(samples) for gesture, samples in dataset.items()},
            "feature_dimension": None
        }
        
        # Get feature dimension from first sample
        if dataset:
            first_samples = next(iter(dataset.values()))
            if first_samples:
                stats["feature_dimension"] = len(first_samples[0])
        
        return stats
    
    def interactive_collection(self):
        """Interactive gesture collection interface"""
        print("=== Gesture Dataset Collector ===")
        print("Available gestures:", self.get_available_gestures())
        
        while True:
            print("\nOptions:")
            print("1. Collect new gesture")
            print("2. Add samples to existing gesture")
            print("3. View dataset statistics")
            print("4. View gesture-command mappings")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                gesture_name = input("Enter new gesture name: ").strip().lower()
                if not gesture_name:
                    print("Invalid gesture name")
                    continue
                
                # Ask for command mapping
                print("\nCommon commands: hello, time, stop, play_music, pause_music, volume_up, volume_down, weather, news, search, help")
                command = input(f"Enter command for gesture '{gesture_name}': ").strip().lower()
                if not command:
                    print("Command is required. Using gesture name as command.")
                    command = gesture_name
                
                # Save metadata before collecting samples
                self.save_gesture_metadata(gesture_name, command)
                
                target_count = int(input(f"Number of samples (default {SAMPLES_PER_GESTURE}): ").strip() 
                                 or str(SAMPLES_PER_GESTURE))
                
                self.collect_gesture_samples(gesture_name, target_count)
                
            elif choice == '2':
                gestures = self.get_available_gestures()
                if not gestures:
                    print("No gestures found. Collect new gestures first.")
                    continue
                
                print("Available gestures:", gestures)
                gesture_name = input("Enter gesture name: ").strip().lower()
                
                if gesture_name not in gestures:
                    print("Gesture not found")
                    continue
                
                # Check if gesture has metadata, if not ask for command
                metadata = self.load_gesture_metadata(gesture_name)
                if not metadata or "command" not in metadata:
                    print(f"Gesture '{gesture_name}' has no command mapping.")
                    command = input(f"Enter command for gesture '{gesture_name}': ").strip().lower()
                    if not command:
                        command = gesture_name
                    self.save_gesture_metadata(gesture_name, command)
                else:
                    print(f"Gesture '{gesture_name}' is mapped to command: {metadata['command']}")
                
                target_count = int(input(f"Number of samples to add: ").strip())
                self.collect_gesture_samples(gesture_name, target_count)
                
            elif choice == '3':
                stats = self.get_dataset_statistics()
                print("\n=== Dataset Statistics ===")
                print(f"Total gestures: {stats['total_gestures']}")
                print(f"Total samples: {stats['total_samples']}")
                print(f"Feature dimension: {stats['feature_dimension']}")
                print("Samples per gesture:")
                for gesture, count in stats['gesture_counts'].items():
                    print(f"  {gesture}: {count}")
                    
            elif choice == '4':
                mappings = self.load_all_gesture_mappings()
                print("\n=== Gesture-Command Mappings ===")
                if mappings:
                    for gesture, command in mappings.items():
                        print(f"  {gesture}: {command}")
                else:
                    print("No mappings found. Collect gestures with commands first.")
                    
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice")

# Utility function for testing
def test_dataset_collector():
    """Test the dataset collector"""
    collector = GestureDatasetCollector()
    collector.interactive_collection()

if __name__ == "__main__":
    test_dataset_collector()
