"""
Rasa Chatbot Engine
Part 5: Conversational AI using Rasa
"""
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChatMessage:
    text: str
    sender: str  # "user" or "bot"
    timestamp: datetime
    confidence: float = 1.0

class RasaChatbot:
    def __init__(self):
        self.conversation_history = []
        self.is_connected = False
        
        # Check if Rasa is available
        self.check_rasa_connection()
    
    def check_rasa_connection(self):
        """Check if Rasa server is running"""
        try:
            import requests
            response = requests.get("http://localhost:5005/", timeout=2)
            self.is_connected = response.status_code == 200
            if self.is_connected:
                print("Rasa server connected at http://localhost:5005")
            else:
                print("Rasa server not available - using fallback responses")
        except requests.exceptions.ConnectionError:
            self.is_connected = False
            print("Rasa server not running. Start Rasa with: rasa run --enable-api --cors \"*\" --port 5005")
        except Exception as e:
            self.is_connected = False
            print(f"Rasa server not available: {e} - using fallback responses")
    
    def send_message(self, message: str) -> Optional[ChatMessage]:
        """Send message to Rasa and get response"""
        if not self.is_connected:
            return self.get_fallback_response(message)
        
        try:
            import requests
            
            # Send message to Rasa
            payload = {"sender": "user", "message": message}
            response = requests.post(
                "http://localhost:5005/webhooks/rest/webhook",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                responses = response.json()
                if responses:
                    bot_response = responses[0]
                    bot_message = ChatMessage(
                        text=bot_response.get("text", "I didn't understand that."),
                        sender="bot",
                        timestamp=datetime.now(),
                        confidence=bot_response.get("confidence", 1.0)
                    )
                    
                    # Add to conversation history
                    self.conversation_history.append(ChatMessage(
                        text=message,
                        sender="user",
                        timestamp=datetime.now()
                    ))
                    self.conversation_history.append(bot_message)
                    
                    return bot_message
            
        except Exception as e:
            print(f"Error communicating with Rasa: {e}")
            return self.get_fallback_response(message)
        
        return None
    
    def get_fallback_response(self, message: str) -> ChatMessage:
        """Get fallback response when Rasa is not available"""
        # Simple rule-based responses
        message_lower = message.lower()
        
        # Exact command matches for trained gestures - check these first
        if message_lower == "volume_down" or message_lower == "down":
            response = "Volume decreased."
        elif message_lower == "volume up" or message_lower == "up":
            response = "Volume increased."
        elif message_lower == "ok":
            response = "OK"
        elif message_lower == "music is pause" or message_lower == "paper":
            response = "Music paused."
        elif message_lower == "music is played" or message_lower == "rock":
            response = "Music played."
        elif message_lower == "stop":
            response = "Stop"
        elif message_lower == "confirm":
            response = "Confirm"
        elif message_lower == "fist":
            response = "Stop"
        # General command matches
        elif "hello" in message_lower or "hi" in message_lower:
            response = "Hello! How can I help you today?"
        elif "time" in message_lower:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}."
        elif "weather" in message_lower:
            response = "I'm sorry, I don't have access to weather information right now."
        elif "help" in message_lower:
            response = "I can help you with basic commands. Try asking about the time, weather, or just say hello!"
        elif "exit" in message_lower:
            response = "Goodbye! Have a great day!"
        elif "play" in message_lower and "music" in message_lower:
            response = "I would play music for you, but I don't have music integration yet."
        elif "pause" in message_lower and "music" in message_lower:
            response = "Music paused."
        elif "volume" in message_lower:
            if "up" in message_lower:
                response = "Volume increased."
            elif "down" in message_lower:
                response = "Volume decreased."
            else:
                response = "Volume adjusted."
        elif "search" in message_lower:
            response = "I would search for that, but I don't have search integration yet."
        elif "call" in message_lower:
            response = "I would make a call, but I don't have calling integration yet."
        elif "message" in message_lower or "text" in message_lower:
            response = "I would send a message, but I don't have messaging integration yet."
        elif "navigate" in message_lower:
            response = "I would navigate, but I don't have navigation integration yet."
        elif "settings" in message_lower:
            response = "Settings menu would open here."
        elif "calendar" in message_lower:
            response = "I would show your calendar, but I don't have calendar integration yet."
        elif "reminder" in message_lower:
            response = "I would set a reminder, but I don't have reminder integration yet."
        elif "news" in message_lower:
            response = "I would show you the news, but I don't have news integration yet."
        else:
            # For any other message, just echo it back as confirmation
            response = message
        
        bot_message = ChatMessage(
            text=response,
            sender="bot",
            timestamp=datetime.now(),
            confidence=0.5  # Lower confidence for fallback
        )
        
        # Add to conversation history
        self.conversation_history.append(ChatMessage(
            text=message,
            sender="user",
            timestamp=datetime.now()
        ))
        self.conversation_history.append(bot_message)
        
        return bot_message
    
    def get_conversation_history(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
    
    def export_conversation(self, filename: str):
        """Export conversation to file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "text": msg.text,
                    "sender": msg.sender,
                    "timestamp": msg.timestamp.isoformat(),
                    "confidence": msg.confidence
                }
                for msg in self.conversation_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Conversation exported to {filename}")
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {}
        
        user_messages = [msg for msg in self.conversation_history if msg.sender == "user"]
        bot_messages = [msg for msg in self.conversation_history if msg.sender == "bot"]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages),
            "avg_bot_confidence": sum(msg.confidence for msg in bot_messages) / len(bot_messages) if bot_messages else 0,
            "conversation_start": self.conversation_history[0].timestamp.isoformat(),
            "last_message": self.conversation_history[-1].timestamp.isoformat()
        }

# Utility function for testing
def test_rasa_bot():
    """Test the Rasa chatbot"""
    bot = RasaChatbot()
    
    print("=== Rasa Chatbot Test ===")
    print(f"Rasa connected: {bot.is_connected}")
    
    # Test messages
    test_messages = [
        "Hello",
        "What time is it?",
        "How's the weather?",
        "Play music",
        "Help",
        "Unknown command test"
    ]
    
    print("\nTesting responses:")
    for message in test_messages:
        response = bot.send_message(message)
        print(f"User: {message}")
        print(f"Bot: {response.text} (confidence: {response.confidence:.2f})")
        print()
    
    # Test statistics
    stats = bot.get_conversation_stats()
    print("Conversation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_rasa_bot()
