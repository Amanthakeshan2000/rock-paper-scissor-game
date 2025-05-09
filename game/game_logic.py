import random
from ..utils.constants import *

class GameLogic:
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0
        self.rounds_played = 0
        self.max_rounds = 5
        
    def play_round(self, player_gesture):
        if self.rounds_played >= self.max_rounds:
            return True
            
        computer_gesture = random.choice(GESTURES)
        result = self.determine_winner(player_gesture, computer_gesture)
        
        if result == "player":
            self.player_score += 1
        elif result == "computer":
            self.computer_score += 1
            
        self.rounds_played += 1
        return self.rounds_played >= self.max_rounds
        
    def determine_winner(self, player_gesture, computer_gesture):
        if player_gesture == computer_gesture:
            return "tie"
            
        if computer_gesture in GESTURE_RULES[player_gesture]:
            return "player"
        else:
            return "computer"
            
    def get_results(self):
        return {
            "player_score": self.player_score,
            "computer_score": self.computer_score,
            "rounds_played": self.rounds_played
        }
        
    def reset(self):
        self.player_score = 0
        self.computer_score = 0
        self.rounds_played = 0 