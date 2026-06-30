"""
Gesture to Command Mapping
Part 4: Map gestures to commands/text
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config.settings import DEFAULT_GESTURE_COMMANDS, GESTURE_CONFIDENCE_THRESHOLD

@dataclass
class GestureCommand:
    gesture: str
    command: str
    text: str
    confidence: float
    timestamp: datetime

class GestureToCommandMapper:
    def __init__(self):
        self.gesture_commands = DEFAULT_GESTURE_COMMANDS.copy()
        self.command_history = []
        self.custom_mappings = {}
        
        # Load mappings from dataset first, then custom mappings
        self.load_dataset_mappings()
        self.load_custom_mappings()
    
    def load_dataset_mappings(self):
        """Load gesture-command mappings from dataset"""
        from dataset.collector import GestureDatasetCollector
        collector = GestureDatasetCollector()
        dataset_mappings = collector.load_all_gesture_mappings()
        
        if dataset_mappings:
            # Dataset mappings should override default mappings
            self.gesture_commands.update(dataset_mappings)
            self.custom_mappings.update(dataset_mappings)
            print(f"Loaded {len(dataset_mappings)} gesture-command mappings from dataset")
            print(f"Dataset mappings: {dataset_mappings}")
    
    def load_custom_mappings(self):
        """Load custom gesture-command mappings"""
        mappings_file = "gesture_engine/custom_mappings.json"
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, 'r') as f:
                    custom = json.load(f)
                # Update gesture commands with custom mappings
                self.custom_mappings.update(custom)
                self.gesture_commands.update(custom)
                print(f"Loaded {len(custom)} custom mappings")
            except Exception as e:
                print(f"Error loading custom mappings: {e}")
    
    def save_custom_mappings(self):
        """Save custom gesture-command mappings"""
        mappings_file = "gesture_engine/custom_mappings.json"
        os.makedirs(os.path.dirname(mappings_file), exist_ok=True)
        
        try:
            with open(mappings_file, 'w') as f:
                json.dump(self.custom_mappings, f, indent=2)
            print(f"Saved {len(self.custom_mappings)} custom mappings")
        except Exception as e:
            print(f"Error saving custom mappings: {e}")
    
    def reload_mappings(self):
        """Reload all mappings from dataset and custom file"""
        self.gesture_commands = DEFAULT_GESTURE_COMMANDS.copy()
        self.custom_mappings = {}
        self.load_dataset_mappings()
        self.load_custom_mappings()
        print("All mappings reloaded")
    
    def add_mapping(self, gesture: str, command: str, text: str = None):
        """Add new gesture-command mapping"""
        if text is None:
            text = command
        
        self.custom_mappings[gesture] = command
        self.gesture_commands[gesture] = command
        
        print(f"Added mapping: {gesture} -> {command}")
        self.save_custom_mappings()
    
    def remove_mapping(self, gesture: str):
        """Remove gesture-command mapping"""
        if gesture in self.custom_mappings:
            del self.custom_mappings[gesture]
            if gesture in self.gesture_commands:
                del self.gesture_commands[gesture]
            print(f"Removed mapping for gesture: {gesture}")
            self.save_custom_mappings()
        else:
            print(f"No custom mapping found for gesture: {gesture}")
    
    def map_gesture_to_command(self, gesture: str, confidence: float) -> Optional[GestureCommand]:
        """Map gesture to command with confidence check"""
        if confidence < GESTURE_CONFIDENCE_THRESHOLD:
            return None
        
        if gesture not in self.gesture_commands:
            return None
        
        command = self.gesture_commands[gesture]
        
        # Create command object
        gesture_command = GestureCommand(
            gesture=gesture,
            command=command,
            text=self.get_command_text(command),
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.command_history.append(gesture_command)
        
        # Keep history size manageable
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-500:]
        
        return gesture_command
    
    def get_command_text(self, command: str) -> str:
        """Get human-readable text for command"""
        # If the command is already a descriptive text, return it as is
        if " " in command or len(command) > 20:
            return command
        
        command_texts = {
            "hello": "Hello! How can I help you?",
            "time": "What time is it?",
            "stop": "Stop",
            "ok": "OK",
            "start_listening": "I'm listening now",
            "select": "Select",
            "confirm": "Confirm",
            "play_music": "Play music",
            "pause_music": "Pause music",
            "volume_up": "Volume increased.",
            "volume_down": "Volume decreased.",
            "up": "Volume increased.",
            "down": "Volume decreased.",
            "next_track": "Next track",
            "previous_track": "Previous track",
            "weather": "What's the weather like?",
            "news": "Tell me the news",
            "reminder": "Set a reminder",
            "calendar": "Show my calendar",
            "search": "Search",
            "call": "Make a call",
            "message": "Send a message",
            "navigation": "Navigate",
            "settings": "Open settings",
            "help": "Help",
            "exit": "Exit"
        }
        
        return command_texts.get(command, command)
    
    def get_available_gestures(self) -> List[str]:
        """Get list of available gestures"""
        return list(self.gesture_commands.keys())
    
    def get_gesture_info(self, gesture: str) -> Optional[Dict]:
        """Get information about a specific gesture"""
        if gesture not in self.gesture_commands:
            return None
        
        return {
            "gesture": gesture,
            "command": self.gesture_commands[gesture],
            "text": self.get_command_text(self.gesture_commands[gesture]),
            "is_custom": gesture in self.custom_mappings
        }
    
    def get_all_mappings(self) -> Dict[str, Dict]:
        """Get all gesture-command mappings"""
        mappings = {}
        for gesture in self.gesture_commands:
            mappings[gesture] = self.get_gesture_info(gesture)
        return mappings
    
    def get_command_history(self, limit: int = 10) -> List[GestureCommand]:
        """Get recent command history"""
        return self.command_history[-limit:]
    
    def get_gesture_statistics(self) -> Dict:
        """Get statistics about gesture usage"""
        if not self.command_history:
            return {}
        
        gesture_counts = {}
        confidence_sum = {}
        
        for cmd in self.command_history:
            gesture = cmd.gesture
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                confidence_sum[gesture] = 0
            
            gesture_counts[gesture] += 1
            confidence_sum[gesture] += cmd.confidence
        
        # Calculate average confidence
        gesture_stats = {}
        for gesture in gesture_counts:
            gesture_stats[gesture] = {
                "count": gesture_counts[gesture],
                "avg_confidence": confidence_sum[gesture] / gesture_counts[gesture],
                "command": self.gesture_commands.get(gesture, "Unknown")
            }
        
        return gesture_stats
    
    def export_mappings(self, filename: str):
        """Export mappings to file"""
        export_data = {
            "default_mappings": DEFAULT_GESTURE_COMMANDS,
            "custom_mappings": self.custom_mappings,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Mappings exported to {filename}")
    
    def import_mappings(self, filename: str):
        """Import mappings from file"""
        try:
            with open(filename, 'r') as f:
                import_data = json.load(f)
            
            if "custom_mappings" in import_data:
                self.custom_mappings.update(import_data["custom_mappings"])
                self.gesture_commands.update(self.custom_mappings)
                self.save_custom_mappings()
                print(f"Imported {len(import_data['custom_mappings'])} mappings")
            
        except Exception as e:
            print(f"Error importing mappings: {e}")

# Utility function for testing
def test_gesture_mapper():
    """Test the gesture to command mapper"""
    mapper = GestureToCommandMapper()
    
    print("=== Gesture to Command Mapper Test ===")
    
    # Test default mappings
    print("\nDefault mappings:")
    for gesture, command in mapper.gesture_commands.items():
        print(f"  {gesture}: {command}")
    
    # Test mapping
    print("\nTesting gesture mapping:")
    test_gestures = [
        ("thumbs_up", 0.9),
        ("peace", 0.8),
        ("fist", 0.95),
        ("unknown", 0.7),
        ("thumbs_up", 0.5)  # Low confidence
    ]
    
    for gesture, confidence in test_gestures:
        cmd = mapper.map_gesture_to_command(gesture, confidence)
        if cmd:
            print(f"  {gesture} ({confidence:.2f}) -> {cmd.command}")
        else:
            print(f"  {gesture} ({confidence:.2f}) -> No mapping")
    
    # Test custom mapping
    print("\nAdding custom mapping:")
    mapper.add_mapping("custom_gesture", "custom_command", "This is a custom command")
    
    # Test statistics
    print("\nGesture statistics:")
    stats = mapper.get_gesture_statistics()
    for gesture, stat in stats.items():
        print(f"  {gesture}: {stat['count']} uses, avg confidence: {stat['avg_confidence']:.2f}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_gesture_mapper()
