"""
ML Model Training Pipeline
Part 3: Train gesture classifier
"""
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from dataset.collector import GestureDatasetCollector
from config.settings import (
    MODEL_PATH, MODEL_PATH_H5, LABELS_PATH, VALIDATION_SPLIT,
    EPOCHS, BATCH_SIZE, LEARNING_RATE
)

class GestureModelTrainer:
    def __init__(self):
        self.model = None
        self.classes = None
        self.dataset_collector = GestureDatasetCollector()
        
        # Create model directory
        os.makedirs("model", exist_ok=True)
        
    def prepare_data(self, dataset: dict = None):
        """Prepare training data"""
        if dataset is None:
            dataset = self.dataset_collector.load_dataset()
        
        if not dataset:
            raise ValueError("No dataset found. Collect gestures first.")
        
        # Prepare features and labels
        X, y = [], []
        
        for gesture_name, samples in dataset.items():
            for sample in samples:
                X.append(sample)
                y.append(gesture_name)
        
        X = np.array(X)
        y = np.array(y)
        
        # Get unique classes and encode labels
        self.classes = np.unique(y)
        y_encoded = np.array([np.where(self.classes == label)[0][0] for label in y])
        
        # Split data using numpy
        num_samples = len(X)
        num_test = int(num_samples * VALIDATION_SPLIT)
        num_train = num_samples - num_test
        
        # Stratified split
        indices = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y_encoded[train_indices]
        y_test = y_encoded[test_indices]
        
        # Convert to categorical for TensorFlow
        y_train_cat = to_categorical(y_train, num_classes=len(self.classes))
        y_test_cat = to_categorical(y_test, num_classes=len(self.classes))
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        
        return X_train, X_test, y_train_cat, y_test_cat
    
    def build_model(self, input_dim: int, num_classes: int):
        """Build TensorFlow model"""
        self.model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X_train=None, X_test=None, y_train=None, y_test=None):
        """Train the model. If no data provided, loads and prepares dataset automatically."""
        # Allow being called with no arguments (GUI-friendly)
        if X_train is None:
            print("Loading and preparing dataset automatically...")
            dataset = self.dataset_collector.load_dataset()
            if not dataset:
                raise ValueError("No dataset found. Please collect gesture samples first.")
            X_train, X_test, y_train, y_test = self.prepare_data(dataset)
        
        # Build model
        input_dim = X_train.shape[1]
        num_classes = len(self.classes)
        
        self.build_model(input_dim, num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH_H5, save_best_only=True)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test_labels)
        print(f"\n=== Model Accuracy ===")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Per-class accuracy
        print("\n=== Per-Class Accuracy ===")
        for i, class_name in enumerate(self.classes):
            class_mask = y_test_labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_test_labels[class_mask])
                print(f"{class_name}: {class_acc:.4f}")
        
        return y_pred
    
    def save_model(self):
        """Save trained model and classes"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs("model", exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(MODEL_PATH_H5)
        print(f"TensorFlow model saved to {MODEL_PATH_H5}")
        
        # Save classes
        with open(LABELS_PATH, 'wb') as f:
            pickle.dump(self.classes, f)
        print(f"Classes saved to {LABELS_PATH}")
    
    def load_model(self):
        """Load trained model and classes"""
        # Load TensorFlow model
        if os.path.exists(MODEL_PATH_H5):
            self.model = tf.keras.models.load_model(MODEL_PATH_H5)
            print(f"TensorFlow model loaded from {MODEL_PATH_H5}")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH_H5}")
        
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Classes not found at {LABELS_PATH}")
        
        # Load classes
        with open(LABELS_PATH, 'rb') as f:
            self.classes = pickle.load(f)
        print(f"Classes loaded. Classes: {list(self.classes)}")
        
        return self.model, self.classes
    
    def predict_gesture(self, features: np.ndarray):
        """Predict gesture from features"""
        if self.model is None:
            self.load_model()
        
        # Ensure features are in correct shape
        features = features.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(features)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_gesture = self.classes[predicted_class_idx]
        
        return predicted_gesture, confidence

# Utility function for testing
def test_model_trainer():
    """Test the model trainer"""
    trainer = GestureModelTrainer()
    
    print("=== Gesture Model Trainer ===")
    
    # Load dataset
    dataset = trainer.dataset_collector.load_dataset()
    if not dataset:
        print("No dataset found. Please collect gestures first.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(dataset)
    
    # Train model
    print("\nTraining model...")
    history = trainer.train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    print("\nEvaluating model...")
    trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    test_model_trainer()
