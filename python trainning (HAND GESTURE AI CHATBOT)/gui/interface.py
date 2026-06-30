"""
GUI Interface
Part 7: Full application interface
Fixed bugs:
 - train_model() now called correctly (no args needed)
 - Gesture debounce/cooldown prevents chatbot flooding
 - Camera can run without model (still shows feed)
 - Better threading and error handling
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime

from vision.hand_tracker import HandTracker
from vision.landmark_extractor import LandmarkExtractor

from model.train import GestureModelTrainer
from gesture_engine.gesture_to_command import GestureToCommandMapper
from chatbot.rasa_bot import RasaChatbot
from voice.tts import TTS
from voice.stt import STT
from config.settings import (
    CHATBOT_MODE, WINDOW_WIDTH, WINDOW_HEIGHT,
    GESTURE_COOLDOWN, GESTURE_CONFIDENCE_THRESHOLD
)

class HandGestureChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture AI Chatbot")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#1e1e2e")

        # Initialize components
        self.hand_tracker = HandTracker()
        self.landmark_extractor = LandmarkExtractor()
        self.model_trainer = GestureModelTrainer()

        self.gesture_mapper = GestureToCommandMapper()
        self.tts = TTS()
        self.stt = STT()

        # Initialize chatbot - use RasaChatbot
        self.chatbot = RasaChatbot()

        # State variables
        self.is_running = False
        self.current_gesture = ""
        self.current_confidence = 0.0
        self.camera_thread = None
        self.model_loaded = False

        # Gesture debounce: track last time a gesture command was sent
        self._last_gesture_time = 0.0
        self._last_sent_gesture = ""

        # Try to load model silently
        self._try_load_model()

        self.setup_ui()

    def _try_load_model(self):
        """Try to load an existing model silently."""
        try:
            self.model_trainer.load_model()
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
        except Exception as e:
            self.model_loaded = False
            print(f"Model load error: {e}")

    # ─────────────────────────────────────────────
    # UI SETUP
    # ─────────────────────────────────────────────

    def setup_ui(self):
        """Setup the user interface."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background="#1e1e2e", foreground="#89b4fa")
        style.configure("TButton", background="#313244", foreground="#cdd6f4", font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#45475a")])

        # Main container
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # ── LEFT PANEL ──────────────────────────────────
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=3)

        # Camera display canvas
        self.camera_canvas = tk.Canvas(
            left_panel, width=320, height=240,
            bg="#11111b", highlightthickness=1,
            highlightbackground="#313244"
        )
        self.camera_canvas.grid(row=0, column=0, sticky="nsew")
        self.camera_canvas.create_text(
            160, 120, text="📷 Camera Feed\nPress Start",
            fill="#6c7086", font=("Segoe UI", 11), justify="center"
        )

        # Model status badge
        model_color = "#a6e3a1" if self.model_loaded else "#f38ba8"
        model_text  = "✓ Model Loaded" if self.model_loaded else "✗ No Model"
        self.model_status_label = tk.Label(
            left_panel, text=model_text, bg="#1e1e2e",
            fg=model_color, font=("Segoe UI", 9, "bold")
        )
        self.model_status_label.grid(row=1, column=0, pady=(4, 0))

        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=2, column=0, pady=6)

        self.start_button = tk.Button(
            control_frame, text="▶ Start", command=self.toggle_camera,
            bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2"
        )
        self.start_button.grid(row=0, column=0, padx=3)

        tk.Button(
            control_frame, text="🔧 Train Model", command=self.train_model,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=4, cursor="hand2"
        ).grid(row=0, column=1, padx=3)

        tk.Button(
            control_frame, text="📊 Collect Data", command=self.collect_data,
            bg="#cba6f7", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=4, cursor="hand2"
        ).grid(row=0, column=2, padx=3)

        # Gesture info panel
        gesture_frame = ttk.LabelFrame(left_panel, text="Gesture Detection", padding=6)
        gesture_frame.grid(row=3, column=0, sticky="ew", pady=6)
        gesture_frame.columnconfigure(0, weight=1)

        self.gesture_label = ttk.Label(gesture_frame, text="Gesture : None", font=("Segoe UI", 9))
        self.gesture_label.grid(row=0, column=0, sticky="w")

        self.confidence_label = ttk.Label(gesture_frame, text="Confidence: —", font=("Segoe UI", 9))
        self.confidence_label.grid(row=1, column=0, sticky="w")

        self.command_label = ttk.Label(gesture_frame, text="Command  : —", font=("Segoe UI", 9))
        self.command_label.grid(row=2, column=0, sticky="w")

        # Confidence bar
        self.conf_bar = ttk.Progressbar(gesture_frame, length=200, maximum=100, mode="determinate")
        self.conf_bar.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        # ── RIGHT PANEL ──────────────────────────────────
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        # Title
        tk.Label(
            right_panel, text="🤖 AI Chatbot",
            bg="#1e1e2e", fg="#89b4fa",
            font=("Segoe UI", 13, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            right_panel, wrap=tk.WORD, state=tk.DISABLED,
            bg="#11111b", fg="#cdd6f4", font=("Segoe UI", 10),
            insertbackground="white", relief="flat",
            selectbackground="#313244"
        )
        self.chat_display.grid(row=1, column=0, sticky="nsew")
        self.chat_display.tag_configure("user", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_configure("bot",  foreground="#a6e3a1", font=("Segoe UI", 10))
        self.chat_display.tag_configure("system", foreground="#6c7086", font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_configure("gesture", foreground="#f9e2af", font=("Segoe UI", 10))

        # Input frame
        input_frame = ttk.Frame(right_panel)
        input_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        input_frame.columnconfigure(0, weight=1)

        self.input_entry = tk.Entry(
            input_frame, bg="#313244", fg="#cdd6f4",
            font=("Segoe UI", 11), relief="flat",
            insertbackground="white"
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 6))
        self.input_entry.bind("<Return>", self.send_text_message)
        self.input_entry.insert(0, "Type a message...")
        self.input_entry.bind("<FocusIn>",  lambda e: self._clear_placeholder())
        self.input_entry.bind("<FocusOut>", lambda e: self._restore_placeholder())

        tk.Button(
            input_frame, text="Send ➤", command=self.send_text_message,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=6, cursor="hand2"
        ).grid(row=0, column=1)

        tk.Button(
            input_frame, text="🎙 Voice", command=self.voice_input,
            bg="#fab387", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=6, cursor="hand2"
        ).grid(row=0, column=2, padx=(6, 0))

        # ── STATUS BAR ──────────────────────────────────
        self.status_label = tk.Label(
            self.root, text="Ready — Use Start to begin camera",
            bg="#181825", fg="#6c7086",
            font=("Segoe UI", 8), anchor="w", padx=8
        )
        self.status_label.grid(row=1, column=0, sticky="ew")

        # Welcome message
        self.add_chat_message(
            "Welcome! I'm your Hand Gesture AI Chatbot. "
            "Start the camera, show gestures or type a message below.",
            "system"
        )

    # ─────────────────────────────────────────────
    # PLACEHOLDER HELPERS
    # ─────────────────────────────────────────────

    def _clear_placeholder(self):
        if self.input_entry.get() == "Type a message...":
            self.input_entry.delete(0, tk.END)
            self.input_entry.config(fg="#cdd6f4")

    def _restore_placeholder(self):
        if not self.input_entry.get():
            self.input_entry.insert(0, "Type a message...")
            self.input_entry.config(fg="#6c7086")

    # ─────────────────────────────────────────────
    # CAMERA CONTROL
    # ─────────────────────────────────────────────

    def toggle_camera(self):
        """Toggle camera on/off."""
        if self.is_running:
            self.stop_camera()
            self.start_button.config(text="▶ Start", bg="#a6e3a1")
        else:
            self.start_camera()
            self.start_button.config(text="■ Stop", bg="#f38ba8")

    def start_camera(self):
        """Start camera. Works even without a trained model (gesture detection disabled)."""
        if not self.hand_tracker.initialize_camera():
            messagebox.showerror("Camera Error", "Failed to initialize camera.\nCheck your webcam connection.")
            return

        if not self.model_loaded:
            self.add_chat_message(
                "⚠ No model found — camera running but gesture detection is disabled. "
                "Use 'Train Model' to enable gestures.",
                "system"
            )

        self.is_running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.update_status("Camera running — show hand gestures")

    def stop_camera(self):
        """Stop camera."""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        self.hand_tracker.release()
        # Clear canvas
        self.camera_canvas.delete("all")
        self.camera_canvas.create_text(
            160, 120, text="📷 Camera Feed\nPress Start",
            fill="#6c7086", font=("Segoe UI", 11), justify="center"
        )
        self.update_status("Camera stopped")

    def camera_loop(self):
        """Main camera processing loop — runs in background thread."""
        while self.is_running:
            frame = self.hand_tracker.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Detect hands
            hand_landmarks, annotated_frame = self.hand_tracker.detect_hands(frame)

            # Gesture recognition (only when model is loaded)
            if hand_landmarks and self.model_loaded:
                # IMPORTANT: must match training feature dimension
                # Training uses LandmarkExtractor.process_landmarks() (not HandTracker.extract_landmark_features)
                try:
                    features = self.landmark_extractor.process_landmarks(hand_landmarks[0])
                except Exception as e:
                    print(f"[ERROR] Feature extraction failed: {e}")
                    features = None
                
                if features is not None:

                    try:
                        # Guard against feature-dimension mismatches
                        expected_dim = getattr(getattr(self.model_trainer, "model", None), "n_features_in_", None)
                        if expected_dim is not None and len(features) != expected_dim:
                            print(
                                f"[Feature Dim Mismatch] got={len(features)} expected={expected_dim}. "
                                f"Did you retrain the model after feature changes?"
                            )
                            return

                        predicted_gesture, confidence = self.model_trainer.predict_gesture(features)

                        # Map to command only if confidence is sufficient
                        gesture_command = self.gesture_mapper.map_gesture_to_command(
                            predicted_gesture, confidence
                        )

                        # Always show prediction results in the UI
                        self.root.after(0, self.update_gesture_info, predicted_gesture, confidence, gesture_command)

                        if gesture_command:
                            # ── DEBOUNCE: prevent flooding chatbot ──
                            now = time.time()
                            same_gesture = (predicted_gesture == self._last_sent_gesture)
                            cooldown_elapsed = (now - self._last_gesture_time) >= GESTURE_COOLDOWN

                            if cooldown_elapsed or not same_gesture:
                                self._last_gesture_time = now
                                self._last_sent_gesture = predicted_gesture
                                self.root.after(0, self.send_gesture_command, gesture_command)

                    except Exception as e:
                        print(f"Gesture prediction error: {e}")

            # Update camera display on main thread
            self.root.after(0, self.update_camera_display, annotated_frame)
            time.sleep(0.033)  # ~30 FPS cap

    # ─────────────────────────────────────────────
    # DISPLAY UPDATES
    # ─────────────────────────────────────────────

    def update_camera_display(self, frame):
        """Update camera canvas with new frame."""
        if not self.is_running or frame is None:
            return
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 240))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(0, 0, anchor="nw", image=photo)
            self.camera_canvas.image = photo  # prevent GC
        except Exception:
            pass

    def update_gesture_info(self, gesture: str, confidence: float, gesture_command=None):
        """Update gesture info labels and confidence bar."""
        self.gesture_label.config(
            text=f"Gesture : {gesture}",
            foreground="#f9e2af"
        )
        conf_pct = confidence * 100
        self.confidence_label.config(
            text=f"Confidence: {conf_pct:.1f}%",
            foreground="#a6e3a1" if conf_pct >= 80 else "#fab387"
        )
        if gesture_command:
            self.command_label.config(text=f"Command  : {gesture_command.command}")
        else:
            self.command_label.config(text="Command  : No command configured")
        self.conf_bar["value"] = conf_pct

    # ─────────────────────────────────────────────
    # MESSAGING
    # ─────────────────────────────────────────────

    def send_gesture_command(self, gesture_command):
        """Send gesture command to chatbot."""
        self.add_chat_message(f"[👋 Gesture: {gesture_command.gesture}] → {gesture_command.text}", "gesture")

        def respond():
            try:
                response = self.chatbot.send_message(gesture_command.text)
                if response:
                    self.root.after(0, self.add_chat_message, response.text, "bot")
                    threading.Thread(target=self.tts.speak, args=(response.text,), daemon=True).start()
            except Exception as e:
                self.root.after(0, self.add_chat_message, f"Error: {e}", "system")

        threading.Thread(target=respond, daemon=True).start()

    def send_text_message(self, event=None):
        """Send text message to chatbot."""
        message = self.input_entry.get().strip()
        if not message or message == "Type a message...":
            return

        self.input_entry.delete(0, tk.END)
        self.add_chat_message(message, "user")
        self.update_status("Thinking...")

        def respond():
            try:
                response = self.chatbot.send_message(message)
                if response:
                    self.root.after(0, self.add_chat_message, response.text, "bot")
                    self.root.after(0, self.update_status, "Ready")
                    threading.Thread(target=self.tts.speak, args=(response.text,), daemon=True).start()
            except Exception as e:
                self.root.after(0, self.add_chat_message, f"Error: {e}", "system")
                self.root.after(0, self.update_status, "Error occurred")

        threading.Thread(target=respond, daemon=True).start()

    def voice_input(self):
        """Handle voice input in background."""
        # STT may be unavailable if PyAudio is not installed.
        if not getattr(self.stt, "is_available", False):
            self.update_status("🎙 Microphone unavailable — install PyAudio")
            self.add_chat_message(
                "🎙 Microphone unavailable. Install PyAudio to enable voice input.",
                "system",
            )
            return
        self.update_status("🎙 Listening...")


        def listen_thread():
            try:
                text = self.stt.listen()
                if text:
                    self.root.after(0, self._process_voice_input, text)
                else:
                    self.root.after(0, self.update_status, "No speech detected")
            except Exception as e:
                self.root.after(0, self.update_status, f"Voice error: {e}")

        threading.Thread(target=listen_thread, daemon=True).start()

    def _process_voice_input(self, text):
        """Process transcribed voice input."""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, text)
        self.input_entry.config(fg="#cdd6f4")
        self.send_text_message()

    def add_chat_message(self, message: str, sender: str):
        """Add styled message to chat display."""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M")

        if sender == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "bot":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] Bot: ", "bot")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "gesture":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] {message}\n", "gesture")
        else:
            self.chat_display.insert(tk.END, f"\n[{timestamp}] {message}\n", "system")

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────
    # MODEL / DATA ACTIONS
    # ─────────────────────────────────────────────

    def train_model(self):
        """Train gesture model in background thread."""
        dataset = self.model_trainer.dataset_collector.load_dataset()
        if not dataset:
            messagebox.showwarning(
                "No Dataset",
                "No gesture data found!\n\nPlease collect gesture samples first using 'Collect Data'."
            )
            return

        if not messagebox.askyesno(
            "Train Model",
            f"Found {len(dataset)} gesture(s) in dataset.\nThis will train a new model. Continue?"
        ):
            return

        def train_thread():
            try:
                self.root.after(0, self.update_status, "Training model... please wait")
                self.root.after(0, self.add_chat_message, "Starting model training...", "system")

                # train_model() with no args: auto-loads data internally (FIXED)
                self.model_trainer.train_model()
                self.model_trainer.save_model()
                self.model_loaded = True

                self.root.after(0, self.update_status, "Model training completed ✓")
                self.root.after(0, self.model_status_label.config,
                                dict(text="✓ Model Loaded", fg="#a6e3a1"))
                self.root.after(0, self.add_chat_message,
                                "Model training completed! Gestures are now active.", "system")
                self.root.after(0, messagebox.showinfo, "Success", "Model trained and saved successfully!")
            except Exception as e:
                self.root.after(0, self.update_status, "Training failed")
                self.root.after(0, self.add_chat_message, f"Training error: {e}", "system")
                self.root.after(0, messagebox.showerror, "Training Error", f"Model training failed:\n{e}")

        threading.Thread(target=train_thread, daemon=True).start()

    def collect_data(self):
        """Launch data collection in a separate thread."""
        from dataset.collector import GestureDatasetCollector

        def collect_thread():
            try:
                collector = GestureDatasetCollector()
                collector.interactive_collection()
            except Exception as e:
                self.root.after(0, messagebox.showerror,
                                "Collection Error", f"Data collection failed:\n{e}")

        threading.Thread(target=collect_thread, daemon=True).start()
        self.add_chat_message(
            "Data collection window opened in terminal. Follow on-screen instructions.", "system"
        )

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def update_status(self, message: str):
        """Update footer status bar."""
        self.status_label.config(text=f"  {message}")

    def on_closing(self):
        """Handle window closing gracefully."""
        self.stop_camera()
        try:
            self.tts.stop()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    root.resizable(True, True)
    app = HandGestureChatbotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
