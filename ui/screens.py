import cv2
import numpy as np
from ..utils.constants import *
from ..utils.image_utils import *

class WelcomeScreen:
    def draw(self, frame):
        # Draw welcome screen
        cv2.putText(frame, "Welcome to Rock Paper Scissors Lizard Spock!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Show your hand to start!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

class GameScreen:
    def draw(self, frame):
        # Draw game screen
        cv2.putText(frame, "Make your move!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

class ResultScreen:
    def draw(self, frame, results):
        # Draw result screen
        cv2.putText(frame, f"Score: {results['player_score']} - {results['computer_score']}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Show your hand to continue!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

class ReportScreen:
    def draw(self, frame):
        # Draw report screen
        cv2.putText(frame, "Generating report...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Show your hand to return to welcome screen!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame 