import cv2
import numpy as np
import time
from ..utils.constants import *
from ..utils.image_utils import *
from ..gesture.gesture_processor import GestureProcessor
from ..ui.screens import WelcomeScreen, GameScreen, ResultScreen, ReportScreen
from ..game.game_logic import GameLogic
from ..report.report_generator import ReportGenerator

class GameEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.gesture_processor = GestureProcessor()
        self.game_logic = GameLogic()
        self.report_generator = ReportGenerator()
        
        self.current_screen = "welcome"
        self.screens = {
            "welcome": WelcomeScreen(),
            "game": GameScreen(),
            "result": ResultScreen(),
            "report": ReportScreen()
        }
        
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # 1 second cooldown between gestures
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Process current screen
            if self.current_screen == "welcome":
                frame = self.screens["welcome"].draw(frame)
                gesture = self.gesture_processor.process_gesture(frame)
                if gesture:
                    self.current_screen = "game"
                    
            elif self.current_screen == "game":
                frame = self.screens["game"].draw(frame)
                gesture = self.gesture_processor.process_gesture(frame)
                if gesture and time.time() - self.last_gesture_time > self.gesture_cooldown:
                    self.last_gesture_time = time.time()
                    result = self.game_logic.play_round(gesture)
                    if result:
                        self.current_screen = "result"
                        
            elif self.current_screen == "result":
                frame = self.screens["result"].draw(frame, self.game_logic.get_results())
                gesture = self.gesture_processor.process_gesture(frame)
                if gesture:
                    self.current_screen = "report"
                    
            elif self.current_screen == "report":
                frame = self.screens["report"].draw(frame)
                gesture = self.gesture_processor.process_gesture(frame)
                if gesture:
                    self.report_generator.generate_report(self.game_logic.get_results())
                    self.current_screen = "welcome"
                    self.game_logic.reset()
            
            cv2.imshow('Rock Paper Scissors Lizard Spock', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows() 