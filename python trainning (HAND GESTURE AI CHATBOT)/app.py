"""
Main Application Runner
Hand Gesture Driven Conversational AI Chatbot
"""
import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Unicode safe print wrapper
def safe_print(text):
    """Safely print text with Unicode fallback"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace common emoji with ASCII equivalents
        safe_text = text.replace('❌', '[X]').replace('✅', '[OK]').replace('🚀', '[>]').replace('👋', '[wave]').replace('🧠', '[brain]').replace('📊', '[chart]').replace('🧪', '[test]').replace('🖐️', '[hand]').replace('→', '->')
        print(safe_text)

from gui.interface import main as gui_main
from dataset.collector import GestureDatasetCollector
from model.train import GestureModelTrainer
from vision.hand_tracker import test_hand_tracker
from voice.tts import TTS
from voice.stt import STT
from config.settings import MODEL_PATH, MODEL_PATH_H5

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🖐️  Hand Gesture Driven Conversational AI Chatbot      ║
    ║                                                              ║
    ║    Camera → Hand Gesture → AI Model → Command/Text → Chatbot ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Unicode
        simple_banner = """
    ============================================================
    
           Hand Gesture Driven Conversational AI Chatbot
    
        Camera -> Hand Gesture -> AI Model -> Command/Text -> Chatbot
    
    ============================================================
        """
        print(simple_banner)

def check_dependencies():
    """Check if required dependencies are installed.

    NOTE: mediapipe is checked via importlib.util.find_spec (NOT __import__)
    because on Python 3.13+ loading mediapipe's C-extension hangs indefinitely.
    """
    import importlib.util

    # Core required modules — safe to import
    required_modules = [
        'cv2', 'numpy', 'pyttsx3', 'speech_recognition'
    ]

    # Optional modules — use spec check to avoid import hangs
    optional_modules = ['mediapipe', 'tensorflow']

    missing_required = []
    missing_optional = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)

    for module in optional_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                missing_optional.append(module)
        except (ModuleNotFoundError, ValueError):
            missing_optional.append(module)

    if missing_required:
        safe_print(f"❌ Missing required dependencies: {', '.join(missing_required)}")
        safe_print("Please install required packages using:")
        safe_print("pip install -r requirements.txt")
        return False

    if missing_optional:
        safe_print(f"[WARNING] Missing optional dependencies: {', '.join(missing_optional)}")
        safe_print("[WARNING] Hand gesture detection and AI model training will be limited")
        safe_print("[WARNING] Install mediapipe and tensorflow for full functionality")
    else:
        # Note: mediapipe may be installed but unsupported on this Python version
        if sys.version_info >= (3, 13):
            safe_print("[WARNING] mediapipe found but Python 3.13+ is not officially supported.")
            safe_print("[WARNING] Running in OpenCV-only hand detection mode.")
        else:
            safe_print("✅ All dependencies are installed")

    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'dataset/samples',
        'model',
        'gesture_engine',
        'chatbot',
        'voice',
        'gui',
        'utils',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    safe_print("✅ Directory structure created")

def run_gui():
    """Run the GUI application"""
    safe_print("🚀 Starting GUI Application...")
    try:
        gui_main()
    except KeyboardInterrupt:
        safe_print("\n👋 Application stopped by user")
    except Exception as e:
        safe_print(f"❌ Error running GUI: {e}")

def collect_data():
    """Run data collection"""
    safe_print("📊 Starting Data Collection...")
    collector = GestureDatasetCollector()
    collector.interactive_collection()

def train_model():
    """Train the gesture model"""
    safe_print("🧠 Starting Model Training...")
    trainer = GestureModelTrainer()
    
    # Load dataset
    dataset = trainer.dataset_collector.load_dataset()
    if not dataset:
        safe_print("❌ No dataset found. Please collect gestures first.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(dataset)
    
    # Train model
    safe_print("Training model...")
    history = trainer.train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    safe_print("Evaluating model...")
    trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    safe_print("✅ Model training completed!")

def test_components():
    """Test individual components"""
    safe_print("🧪 Testing Components...")
    
    safe_print("\n1. Testing Hand Tracker...")
    test_hand_tracker()
    
    safe_print("\n2. Testing TTS...")
    tts = TTS()
    tts.speak("Text to speech test completed")
    
    safe_print("\n3. Testing STT...")
    stt = STT()
    safe_print("Microphone initialized. Speak something to test...")
    # Note: STT test requires user interaction
    
    safe_print("\n✅ Component testing completed")

def show_status():
    """Show application status"""
    safe_print("📊 Application Status")
    safe_print("=" * 50)
    
    # Check model
    if os.path.exists(MODEL_PATH_H5):
        size = os.path.getsize(MODEL_PATH_H5) / (1024 * 1024)  # MB
        safe_print(f"✅ TensorFlow model: {MODEL_PATH_H5} ({size:.1f} MB)")
    elif os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        safe_print(f"✅ Scikit-learn model: {MODEL_PATH} ({size:.1f} MB)")
    else:
        safe_print(f"❌ Model: neither {MODEL_PATH} nor {MODEL_PATH_H5} found")
    
    # Check dataset
    dataset_path = "dataset/samples"
    if os.path.exists(dataset_path):
        gestures = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
        safe_print(f"✅ Dataset: {len(gestures)} gestures collected")
        for gesture in gestures:
            count = len([f for f in os.listdir(os.path.join(dataset_path, gesture)) 
                        if f.endswith('.json')])
            safe_print(f"   - {gesture}: {count} samples")
    else:
        safe_print("❌ Dataset: No gestures collected")
    
    # Check configuration
    config_path = "config/settings.py"
    if os.path.exists(config_path):
        safe_print(f"✅ Configuration: {config_path}")
    else:
        safe_print(f"❌ Configuration: {config_path} (not found)")
    
    safe_print("=" * 50)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Hand Gesture Driven Conversational AI Chatbot"
    )
    parser.add_argument(
        '--mode', 
        choices=['gui', 'collect', 'train', 'test', 'status'],
        default='gui',
        help='Application mode (default: gui)'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies only'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            safe_print("✅ All dependencies satisfied")
        else:
            safe_print("❌ Please install missing dependencies")
            sys.exit(1)
        return
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run based on mode
    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'collect':
        collect_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_components()
    elif args.mode == 'status':
        show_status()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\n👋 Goodbye!")
    except Exception as e:
        safe_print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
