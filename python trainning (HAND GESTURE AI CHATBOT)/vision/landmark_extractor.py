"""
Landmark Extractor Module
Part 1: Extract and process hand landmarks for gesture recognition
"""
import numpy as np
from typing import List, Optional, Tuple

class LandmarkExtractor:
    def __init__(self):
        # MediaPipe hand landmark indices
        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_FINGER_MCP = 5
        self.INDEX_FINGER_PIP = 6
        self.INDEX_FINGER_DIP = 7
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_MCP = 9
        self.MIDDLE_FINGER_PIP = 10
        self.MIDDLE_FINGER_DIP = 11
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_MCP = 13
        self.RING_FINGER_PIP = 14
        self.RING_FINGER_DIP = 15
        self.RING_FINGER_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position
        Args:
            landmarks: Array of shape (21, 3) for x, y, z coordinates
        Returns:
            Normalized landmarks array
        """
        if landmarks.shape != (21, 3):
            raise ValueError("Landmarks must have shape (21, 3)")
        
        # Translate so wrist is at origin
        wrist_position = landmarks[self.WRIST]
        normalized = landmarks - wrist_position
        
        # Scale to unit distance (using middle finger tip as reference)
        reference_distance = np.linalg.norm(
            normalized[self.MIDDLE_FINGER_TIP] - normalized[self.WRIST]
        )
        if reference_distance > 0:
            normalized = normalized / reference_distance
        
        return normalized
    
    def extract_finger_states(self, landmarks: np.ndarray) -> dict:
        """
        Extract finger states (extended/folded) from landmarks
        Args:
            landmarks: Normalized landmarks array
        Returns:
            Dictionary with finger states
        """
        finger_states = {}
        
        # Thumb: Check if tip is extended compared to IP joint
        thumb_extended = landmarks[self.THUMB_TIP][0] > landmarks[self.THUMB_IP][0]
        finger_states['thumb'] = int(thumb_extended)
        
        # Other fingers: Check if tip is extended compared to PIP joint
        fingers = [
            ('index', self.INDEX_FINGER_TIP, self.INDEX_FINGER_PIP),
            ('middle', self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_PIP),
            ('ring', self.RING_FINGER_TIP, self.RING_FINGER_PIP),
            ('pinky', self.PINKY_TIP, self.PINKY_PIP)
        ]
        
        for name, tip, pip in fingers:
            # Finger is extended if tip is further from wrist than PIP
            tip_distance = np.linalg.norm(landmarks[tip])
            pip_distance = np.linalg.norm(landmarks[pip])
            extended = tip_distance > pip_distance
            finger_states[name] = int(extended)
        
        return finger_states
    
    def calculate_hand_orientation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate hand orientation using PCA
        Args:
            landmarks: Normalized landmarks array
        Returns:
            Orientation vector
        """
        # Use finger tips for orientation calculation
        finger_tips = [
            landmarks[self.INDEX_FINGER_TIP],
            landmarks[self.MIDDLE_FINGER_TIP],
            landmarks[self.RING_FINGER_TIP],
            landmarks[self.PINKY_TIP]
        ]
        
        finger_tips = np.array(finger_tips)
        
        # Calculate covariance matrix and eigenvectors
        cov_matrix = np.cov(finger_tips.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Return principal component (direction of maximum variance)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        
        return principal_component
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract geometric features for gesture classification
        Args:
            landmarks: Normalized landmarks array
        Returns:
            Feature vector
        """
        features = []
        
        # Finger states
        finger_states = self.extract_finger_states(landmarks)
        features.extend(finger_states.values())
        
        # Distances between fingertips
        fingertip_pairs = [
            (self.INDEX_FINGER_TIP, self.MIDDLE_FINGER_TIP),
            (self.MIDDLE_FINGER_TIP, self.RING_FINGER_TIP),
            (self.RING_FINGER_TIP, self.PINKY_TIP),
            (self.INDEX_FINGER_TIP, self.THUMB_TIP),
            (self.MIDDLE_FINGER_TIP, self.THUMB_TIP),
        ]
        
        for tip1, tip2 in fingertip_pairs:
            distance = np.linalg.norm(landmarks[tip1] - landmarks[tip2])
            features.append(distance)
        
        # Angles between fingers
        finger_vectors = [
            landmarks[self.INDEX_FINGER_TIP] - landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.MIDDLE_FINGER_TIP] - landmarks[self.MIDDLE_FINGER_MCP],
            landmarks[self.RING_FINGER_TIP] - landmarks[self.RING_FINGER_MCP],
            landmarks[self.PINKY_TIP] - landmarks[self.PINKY_MCP],
        ]
        
        # Calculate angles between adjacent fingers
        for i in range(len(finger_vectors) - 1):
            v1, v2 = finger_vectors[i], finger_vectors[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(angle)
        
        # Hand orientation
        orientation = self.calculate_hand_orientation(landmarks)
        features.extend(orientation)
        
        return np.array(features)
    
    def process_landmarks(self, raw_landmarks: List[float]) -> np.ndarray:
        """
        Process raw landmarks into feature vector
        Args:
            raw_landmarks: List of 63 floats (21 landmarks * 3 coordinates)
        Returns:
            Processed feature vector
        """
        if len(raw_landmarks) != 63:
            raise ValueError("Expected 63 landmark values (21 * 3)")
        
        # Reshape to (21, 3)
        landmarks = np.array(raw_landmarks).reshape(21, 3)
        
        # Normalize landmarks
        normalized = self.normalize_landmarks(landmarks)
        
        # Extract geometric features
        features = self.extract_geometric_features(normalized)
        
        # Combine with normalized coordinates
        combined_features = np.concatenate([normalized.flatten(), features])
        
        return combined_features

# Utility function for testing
def test_landmark_extractor():
    """Test the landmark extractor"""
    extractor = LandmarkExtractor()
    
    # Create sample landmarks (mock data)
    sample_landmarks = np.random.rand(21, 3)
    
    print("Testing Landmark Extractor...")
    
    # Test normalization
    normalized = extractor.normalize_landmarks(sample_landmarks)
    print(f"Normalized landmarks shape: {normalized.shape}")
    
    # Test finger states
    finger_states = extractor.extract_finger_states(normalized)
    print(f"Finger states: {finger_states}")
    
    # Test geometric features
    features = extractor.extract_geometric_features(normalized)
    print(f"Geometric features shape: {features.shape}")
    
    # Test complete processing
    raw_landmarks = sample_landmarks.flatten().tolist()
    processed_features = extractor.process_landmarks(raw_landmarks)
    print(f"Processed features shape: {processed_features.shape}")
    
    print("Landmark Extractor test completed successfully!")

if __name__ == "__main__":
    test_landmark_extractor()
