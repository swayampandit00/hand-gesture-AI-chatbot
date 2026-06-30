"""
Speech-to-Text Engine
Part 6: Convert voice to text
"""
import speech_recognition as sr
from typing import Optional

class STT:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_available = False
        self.initialize_microphone()

    def initialize_microphone(self):
        """Initialize microphone.

        If PyAudio is missing (common on fresh installs), STT stays disabled
        but the app must continue running.
        """
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.is_available = True
            print("STT microphone initialized")
        except Exception as e:
            # speech_recognition relies on PyAudio for Microphone.
            msg = str(e)
            if "PyAudio" in msg or "pyaudio" in msg:
                print("Error initializing microphone: Could not find PyAudio. Microphone input disabled.")
            else:
                print(f"Error initializing microphone: {e} — Microphone input disabled")
            self.microphone = None
            self.is_available = False

    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for speech and convert to text."""
        if not self.microphone or not self.is_available:
            print("Microphone not available (PyAudio missing or initialization failed)")
            return None

        
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"STT error: {e}")
            return None
        except Exception as e:
            print(f"STT error: {e}")
            return None

if __name__ == "__main__":
    stt = STT()
    text = stt.listen()
    if text:
        print(f"You said: {text}")
    else:
        print("No speech detected")
