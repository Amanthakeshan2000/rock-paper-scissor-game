# Game constants
GESTURES = ["rock", "paper", "scissors", "lizard", "spock"]

# Game rules
GESTURE_RULES = {
    "rock": ["scissors", "lizard"],
    "paper": ["rock", "spock"],
    "scissors": ["paper", "lizard"],
    "lizard": ["paper", "spock"],
    "spock": ["rock", "scissors"]
}

# UI constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CIRCLE_RADIUS = 100 