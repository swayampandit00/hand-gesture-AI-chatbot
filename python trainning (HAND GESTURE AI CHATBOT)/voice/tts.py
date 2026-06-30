"""
Text-to-Speech Engine
Part 6: Convert text responses to voice
"""
import pyttsx3
import threading
from typing import Optional
from config.settings import VOICE_RATE, VOICE_VOLUME

class TTS:
    def __init__(self):
        self.engine = None
        self.is_initialized = False
        self.speak_lock = threading.Lock()
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize TTS engine"""
        try:
            # Try to initialize pyttsx3 with driver detection
            self.engine = pyttsx3.init(driverName=None)  # Auto-detect driver
            
            # Set properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.engine.setProperty('rate', VOICE_RATE)
            self.engine.setProperty('volume', VOICE_VOLUME)
            self.is_initialized = True
            print("TTS engine initialized successfully")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            print("TTS will be disabled. Voice output unavailable.")
            self.is_initialized = False
    
    def speak(self, text: str, block: bool = True):
        """Convert text to speech"""
        if not self.is_initialized:
            print(f"[TTS disabled] Bot response: {text}")
            return False
        
        if not text or not text.strip():
            return False
        
        try:
            with self.speak_lock:
                # Re-initialize engine if needed (pyttsx3 can become stale)
                try:
                    self.engine.endLoop()
                except:
                    pass
                
                self.engine.say(text)
                
                if block:
                    self.engine.runAndWait()
                else:
                    # Run in separate thread for non-blocking
                    thread = threading.Thread(target=self.engine.runAndWait)
                    thread.daemon = True
                    thread.start()
                
                return True
        except RuntimeError as e:
            # Engine loop error - try to reinitialize
            print(f"TTS engine error, reinitializing: {e}")
            self.initialize_engine()
            if self.is_initialized:
                return self.speak(text, block)
            return False
        except Exception as e:
            print(f"TTS error: {e}")
            return False
    
    def stop(self):
        """Stop speech"""
        if self.is_initialized and self.engine:
            try:
                with self.speak_lock:
                    self.engine.stop()
                    try:
                        self.engine.endLoop()
                    except:
                        pass
            except Exception as e:
                print(f"Error stopping TTS: {e}")

if __name__ == "__main__":
    tts = TTS()
    if tts.is_initialized:
        print("Testing TTS...")
        tts.speak("Hello, this is a test of the text to speech system.")
        print("TTS test completed")
    else:
        print("TTS initialization failed")
