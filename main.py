import cv2
import numpy as np
import random
import time
import os
import sys
from collections import deque
from enum import Enum
import mediapipe as mp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

class Gesture(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    SPOCK = 4

class GameState(Enum):
    WELCOME = 0
    WAITING = 1
    COUNTDOWN = 2
    PLAYING = 3
    RESULT = 4

class HandGestureRecognizer:
    def __init__(self):
        self.cap = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.kernel = np.ones((3, 3), np.uint8)
        self.gesture_history = deque(maxlen=5)
        
    def start_camera(self, camera_index=0):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
        return self.cap
    
    def release_camera(self):
        """Release the camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for hand detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imshow("Blurred", blurred)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)
        cv2.imshow("Background Subtraction", fg_mask)
        
        # Apply thresholding
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresholded", thresh)
        
        # Perform morphological operations
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
        dilated = cv2.dilate(opening, self.kernel, iterations=3)
        cv2.imshow("Morphology", dilated)
        
        return dilated
    
    def find_contours(self, processed_frame):
        """Find contours in the processed frame"""
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (should be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000:  # Minimum area to avoid noise
                return max_contour
        return None
    
    def draw_contour(self, frame, contour):
        """Draw contour on the frame"""
        if contour is not None:
            # Draw contour with red border
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
            
            return hull
        return None
    
    def find_defects(self, contour):
        """Find convexity defects in the hand contour"""
        if contour is None:
            return None, None
        
        hull = cv2.convexHull(contour, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contour, hull)
            return defects, hull
        except:
            return None, hull
    
    def count_fingers(self, contour, defects):
        """Count the number of extended fingers using convexity defects"""
        if contour is None or defects is None:
            return 0
        
        finger_count = 1  # Start with 1 for the thumb
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate triangle sides
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            # Apply cosine law to find angle
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If angle is less than 90 degrees, it's likely a finger
            if angle <= np.pi / 2:
                finger_count += 1
        
        return min(finger_count, 5)  # Limit to 5 fingers max
    
    def recognize_gesture_extended(self, finger_count, contour):
        """Recognize the hand gesture based on finger count and contour shape for RPSLS"""
        if contour is None:
            return None
            
        # Basic gestures
        if finger_count <= 1:
            return "rock"
        elif finger_count == 2:
            return "scissors"
        elif finger_count == 3:
            # Try to distinguish between scissors and spock based on finger positions
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # Can be refined with more sophisticated pattern recognition
                return "spock"  # For now, assume 3 fingers is spock
        elif finger_count == 4:
            return "lizard"  # For now, assume 4 fingers is lizard
        elif finger_count >= 5:
            return "paper"
            
        return None
    
    def get_stabilized_gesture(self, gesture):
        """Stabilize gesture recognition using a history buffer"""
        if gesture:
            self.gesture_history.append(gesture)
            
        if len(self.gesture_history) < 3:
            return None
            
        # Return the most common gesture in the history
        from collections import Counter
        counts = Counter(self.gesture_history)
        return counts.most_common(1)[0][0]

class RockPaperScissorsLizardSpockGame:
    def __init__(self):
        self.recognizer = HandGestureRecognizer()
        self.computer_choice = None
        self.player_choice = None
        self.result = None
        self.game_active = False
        self.countdown_start = 0
        self.state = GameState.WELCOME
        self.countdown_val = 3
        self.welcome_animation_time = 0
        self.particles = []
        
        # Initialize processing images dictionary
        self.processing_images = {}
        
        # Threshold settings
        self.threshold_settings = {
            'binary': 127,
            'adaptive': 11,
            'canny_low': 50,
            'canny_high': 150,
            'morph_kernel': 3,
            'blur_kernel': 7
        }
        
        # Button states
        self.button_states = {
            'play': {'hover': False, 'click': False, 'click_time': 0},
            'exit': {'hover': False, 'click': False, 'click_time': 0},
            'instructions': {'hover': False, 'click': False, 'click_time': 0},
            'threshold': {'hover': False, 'click': False, 'click_time': 0},
            'download': {'hover': False, 'click': False, 'click_time': 0}
        }
        
        # Threshold control window properties
        self.threshold_window_name = "Threshold Controls"
        self.show_threshold_controls = False
        
        # Load gesture images
        self.gesture_images = self.load_gesture_images()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Mouse position for button interaction
        self.mouse_pos = (0, 0)
        
        # Create assets directory if it doesn't exist
        if not os.path.exists("assets"):
            os.makedirs("assets")
            print("Created 'assets' directory. Please place gesture images there.")
            
        # Define winning rules for RPSLS
        self.rules = {
            "rock": ["scissors", "lizard"],      # Rock crushes scissors and lizard
            "paper": ["rock", "spock"],          # Paper covers rock and disproves spock
            "scissors": ["paper", "lizard"],     # Scissors cut paper and decapitate lizard
            "lizard": ["paper", "spock"],        # Lizard eats paper and poisons spock
            "spock": ["rock", "scissors"]        # Spock vaporizes rock and smashes scissors
        }
        
        # Fun win messages
        self.win_messages = {
            ("rock", "scissors"): "Rock crushes Scissors",
            ("rock", "lizard"): "Rock crushes Lizard",
            ("paper", "rock"): "Paper covers Rock",
            ("paper", "spock"): "Paper disproves Spock",
            ("scissors", "paper"): "Scissors cut Paper",
            ("scissors", "lizard"): "Scissors decapitate Lizard",
            ("lizard", "paper"): "Lizard eats Paper",
            ("lizard", "spock"): "Lizard poisons Spock",
            ("spock", "rock"): "Spock vaporizes Rock",
            ("spock", "scissors"): "Spock smashes Scissors"
        }
        
        # Welcome screen properties
        self.window_name = "Rock Paper Scissors Lizard Spock"
        self.width = 1024
        self.height = 768
        self.frame_count = 0
        self.show_instructions = False
        
        # Instructions text
        self.instructions_text = [
            "HOW TO PLAY:",
            "1. Press SPACE to start a round",
            "2. Show your hand gesture when prompted",
            "3. The computer will randomly choose a gesture",
            "4. Rock beats Scissors, Scissors beats Paper, Paper beats Rock",
            "5. Lizard beats Paper and Spock, Spock beats Rock and Scissors",
            "",
            "CONTROLS:",
            "- SPACE: Start round / Play again",
            "- F: Toggle fullscreen",
            "- P: Toggle processing visualization",
            "- Q or ESC: Quit game"
        ]
        
        # Add match history tracking
        self.match_history = []
        self.total_matches = 0
        self.player_wins = 0
        self.computer_wins = 0
        self.ties = 0

    def load_gesture_images(self):
        """Load gesture images from assets folder"""
        images = {}
        try:
            # Load images from assets folder
            for gesture in Gesture:
                image_path = f"assets/{gesture.name.lower()}.png"
                if os.path.exists(image_path):
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Resize image to a consistent size
                        img = cv2.resize(img, (150, 150))
                        images[gesture] = img
                    else:
                        print(f"Failed to load image: {image_path}")
                else:
                    print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error loading images: {e}")
        return images
    
    def create_particle_effect(self, x, y, color, count=20):
        """Create particle effect at given position"""
        for _ in range(count):
            vx = random.uniform(-3, 3)
            vy = random.uniform(-5, -1)
            size = random.uniform(3, 8)
            lifetime = random.uniform(0.5, 2.0)
            self.particles.append({
                'x': x, 'y': y,
                'vx': vx, 'vy': vy,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'age': 0
            })
    
    def update_particles(self, dt):
        """Update particle positions and lifetimes"""
        new_particles = []
        for p in self.particles:
            p['age'] += dt
            if p['age'] > p['lifetime']:
                continue
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            new_particles.append(p)
        self.particles = new_particles
    
    def draw_particles(self, frame):
        """Draw all particles on the frame"""
        for p in self.particles:
            opacity = 1.0 - (p['age'] / p['lifetime'])
            color = (
                int(p['color'][0] * opacity),
                int(p['color'][1] * opacity),
                int(p['color'][2] * opacity)
            )
            cv2.circle(frame, (int(p['x']), int(p['y'])), int(p['size']), color, -1)
        return frame

    def get_gesture_image(self, gesture):
        # Create a blank image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Draw the gesture
        if gesture == Gesture.ROCK:
            cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)
        elif gesture == Gesture.PAPER:
            cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        elif gesture == Gesture.SCISSORS:
            cv2.line(img, (70, 70), (130, 130), (255, 0, 0), 10)
            cv2.line(img, (130, 70), (70, 130), (255, 0, 0), 10)
        elif gesture == Gesture.LIZARD:
            pts = np.array([[100, 50], [70, 100], [100, 150], [130, 100]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 255, 255))
        elif gesture == Gesture.SPOCK:
            cv2.line(img, (100, 50), (100, 150), (255, 0, 255), 10)
            cv2.line(img, (70, 70), (70, 150), (255, 0, 255), 10)
            cv2.line(img, (130, 70), (130, 150), (255, 0, 255), 10)
        
        cv2.putText(img, gesture.name, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img
    
    def generate_match_report(self):
        """Generate a creative PDF report of the match results"""
        # Create reports directory if it doesn't exist
        if not os.path.exists("reports"):
            os.makedirs("reports")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/match_report_{timestamp}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.HexColor('#2C3E50'),
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor('#34495E'),
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        # Add decorative header with line breaks
        header_text = "ROCK PAPER SCISSORS <br/> Game Play"
        story.append(Paragraph(header_text, title_style))
        
        # Add smaller subtitle
        subtitle_style = ParagraphStyle(
            'SmallSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#7F8C8D'),
            alignment=1,
            fontName='Helvetica'
        )
        story.append(Paragraph("Match Report", subtitle_style))
        story.append(Spacer(1, 20))
        
        # Add match statistics with creative styling
        stats_data = [
            ["Total Matches", str(self.total_matches)],
            ["Player Wins", str(self.player_wins)],
            ["Computer Wins", str(self.computer_wins)],
            ["Ties", str(self.ties)],
            ["Win Rate", f"{(self.player_wins/self.total_matches*100):.1f}%" if self.total_matches > 0 else "0%"]
        ]
        
        # Create a more visually appealing stats table
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#E8F6F3'), colors.HexColor('#FEF9E7')]),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2C3E50')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7'))
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 30))
        
        # Add match history with creative styling
        story.append(Paragraph("Match History", styles['Heading2']))
        
        if self.match_history:
            # Create headers with emojis
            history_data = [["#️Match", "Player", "Computer", "Result"]]
        
            # Add match data with alternating row colors
            for i, match in enumerate(self.match_history, 1):
                # Add emoji based on result
                result_emoji = "" if "Tie" in match['result'] else "🎉" if "You Win" in match['result'] else ""
                history_data.append([
                    str(i),
                    match['player'],
                    match['computer'],
                    f"{result_emoji} {match['result']}"
                ])
            
            # Create a more visually appealing history table
            history_table = Table(history_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 2.5*inch])
            history_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F9FA')]),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2C3E50')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7'))
            ]))
            story.append(history_table)
        else:
            story.append(Paragraph("No matches played yet. Time to start playing!", styles['Normal']))
        
        # Add a fun footer with timestamp and decorative elements
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#7F8C8D'),
            alignment=1
        )
        footer_text = f"✨ Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ✨"
        story.append(Paragraph(footer_text, footer_style))
        
        # Add a fun message based on win rate
        if self.total_matches > 0:
            win_rate = (self.player_wins/self.total_matches*100)
            if win_rate >= 70:
                message = "🌟 You're a Rock Paper Scissors Lizard Spock Master!"
            elif win_rate >= 50:
                message = "🎯 Great job! Keep practicing to improve!"
            else:
                message = "💪 Don't give up! Practice makes perfect!"
            story.append(Spacer(1, 20))
            story.append(Paragraph(message, footer_style))
        
        # Build the PDF
        doc.build(story)
        return filename
    
    def determine_winner(self, player, computer):
        """Determine the winner based on the extended game rules"""
        if player == computer:
            result = "Tie!"
            self.ties += 1
        elif computer in self.rules.get(player, []):
            message = self.win_messages.get((player, computer), "You Win!")
            result = f"You Win! ({message})"
            self.player_wins += 1
        else:
            message = self.win_messages.get((computer, player), "Pi Wins!")
            result = f"Pi Wins! ({message})"
            self.computer_wins += 1
        
        # Update match history
        self.total_matches += 1
        self.match_history.append({
            'player': player.name if player else 'None',
            'computer': computer.name,
            'result': result
        })
        
        return result
    
    def get_computer_choice(self):
        """Generate a random choice for the computer"""
        return random.choice(["rock", "paper", "scissors", "lizard", "spock"])
    
    def overlay_image(self, background, foreground, position, radius=None):
        """Overlay the foreground image on the background at the specified position with optional circular mask"""
        if foreground is None:
            return background
            
        # Resize the foreground image
        foreground = cv2.resize(foreground, (150, 150))
        
        # Get dimensions
        h, w = foreground.shape[:2]
        x, y = position
        
        # Check if position is within frame boundaries
        bg_h, bg_w = background.shape[:2]
        if x + w > bg_w or y + h > bg_h or x < 0 or y < 0:
            return background
            
        # Create circular mask if radius is specified
        if radius is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w//2, h//2), radius, 255, -1)
            
            # Apply mask to foreground
            if foreground.shape[2] == 4:  # If image has alpha channel
                alpha = foreground[:, :, 3] / 255.0
                alpha = alpha * (mask / 255.0)  # Combine with circular mask
                for c in range(3):
                    background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - alpha) + foreground[:, :, c] * alpha
            else:
                for c in range(3):
                    background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask/255.0) + foreground[:, :, c] * (mask/255.0)
        else:
            # Regular overlay without mask
            if foreground.shape[2] == 4:
                alpha = foreground[:, :, 3] / 255.0
                for c in range(3):
                    background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - alpha) + foreground[:, :, c] * alpha
            else:
                background[y:y+h, x:x+w] = foreground
            
        return background
    
    def draw_game_ui(self, frame, player_gesture, computer_gesture, result):
        """Draw the game UI elements on the frame"""
        h, w = frame.shape[:2]
        
        # Add dark overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw title
        cv2.putText(frame, "Rock Paper Scissors Lizard Spock", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw player and computer choices
        if player_gesture:
            cv2.putText(frame, f"Your Move: {player_gesture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if computer_gesture:
            cv2.putText(frame, f"Pi's Move: {computer_gesture}", (w - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw result
        if result:
            # Extract the win message
            win_message = result.split('(')[0].strip()
            result_color = (0, 255, 0) if "You Win" in win_message else (0, 165, 255) if "Tie" in win_message else (0, 0, 255)
            cv2.putText(frame, f"Winner: {win_message}", (w//2 - 150, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
            
            # If there's a detailed message, show it too
            if "(" in result:
                detailed_message = result.split('(')[1].rstrip(')')
                cv2.putText(frame, detailed_message, (w//2 - 150, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # Add gesture images if available
        if player_gesture and player_gesture in self.gesture_images:
            frame = self.overlay_image(frame, self.gesture_images[player_gesture], (50, 100))
            
        if computer_gesture and computer_gesture in self.gesture_images:
            frame = self.overlay_image(frame, self.gesture_images[computer_gesture], (w - 200, 100))
            
        return frame
    
    def draw_rules_reminder(self, frame):
        """Draw a small reminder of the rules"""
        h, w = frame.shape[:2]
        text_y = h - 140
        text_x = 10
        
        cv2.putText(frame, "Rules:", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Rock crushes Scissors & Lizard", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Paper covers Rock & disproves Spock", (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Scissors cut Paper & decapitate Lizard", (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Lizard eats Paper & poisons Spock", (text_x, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Spock vaporizes Rock & smashes Scissors", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_welcome_screen(self, frame):
        """Draw the welcome screen with animations and creative buttons"""
        # Create a dark blue background
        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background[:] = (79, 27, 27)  # Dark blue in BGR
        
        # Add subtle gradient
        for y in range(self.height):
            alpha = y / self.height
            color = (
                int(79 * (1 - alpha) + 50 * alpha),
                int(27 * (1 - alpha) + 20 * alpha),
                int(27 * (1 - alpha) + 50 * alpha)
            )
            background[y, :] = color
        
        # Add title with shadow effect
        title = "ROCK PAPER SCISSORS"
        subtitle = "Game Play"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw title
        title_size = cv2.getTextSize(title, font, 2, 5)[0]
        title_x = (self.width - title_size[0]) // 2
        cv2.putText(background, title, (title_x + 3, 103), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(background, title, (title_x, 100), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
        
        # Draw subtitle with increased bottom margin
        subtitle_size = cv2.getTextSize(subtitle, font, 1.5, 4)[0]
        subtitle_x = (self.width - subtitle_size[0]) // 2
        cv2.putText(background, subtitle, (subtitle_x + 2, 180), font, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(background, subtitle, (subtitle_x, 178), font, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        
        # Draw gesture images from assets
        icon_size = 150
        circle_radius = 70
        total_width = 3 * (2 * circle_radius + 40)  # Width for 3 circles with spacing
        start_x = (self.width - total_width) // 2  # Center the entire gesture section
        icon_y = 300  # Increased from 250 to create more space after subtitle
        
        # Draw each gesture image in a circle
        positions = [
            (start_x + circle_radius + 20, icon_y, Gesture.ROCK, "ROCK"),
            (start_x + 3 * circle_radius + 60, icon_y, Gesture.PAPER, "PAPER"),
            (start_x + 5 * circle_radius + 100, icon_y, Gesture.SCISSORS, "SCISSORS")
        ]
        
        for pos_x, pos_y, gesture, name in positions:
            # Draw outer circle with color
            color = (0, 165, 255) if gesture == Gesture.ROCK else (0, 255, 0) if gesture == Gesture.PAPER else (255, 0, 0)
            cv2.circle(background, (pos_x, pos_y), circle_radius + 10, color, 10)
            
            # Draw white inner circle
            cv2.circle(background, (pos_x, pos_y), circle_radius, (240, 240, 240), -1)
            
            # Overlay gesture image if available
            if gesture in self.gesture_images:
                img = self.gesture_images[gesture]
                # Calculate position to center the image in the circle
                img_x = pos_x - icon_size // 2
                img_y = pos_y - icon_size // 2
                # Overlay the image with circular mask
                background = self.overlay_image(background, img, (img_x, img_y), circle_radius - 5)
            
            # Add name below the circle
            name_size = cv2.getTextSize(name, font, 0.8, 2)[0]
            name_x = pos_x - name_size[0] // 2
            name_y = pos_y + circle_radius + 40
            
            # Draw name with shadow
            cv2.putText(background, name, (name_x + 2, name_y + 2), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(background, name, (name_x, name_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Adjust button positions to maintain proper spacing
        button_width, button_height = 200, 80
        button_x = (self.width - button_width) // 2
        button_y = 500  # Increased from 450 to maintain spacing
        
        # Play button
        play_hover = self.button_states['play']['hover']
        play_click = self.button_states['play']['click']
        button_color = (0, 200, 100) if play_hover else (0, 150, 80)
        button_color2 = (0, 255, 150) if play_hover else (0, 200, 120)
        
        # Draw play button with gradient
        y_offset = 3 if play_click else 0
        for y in range(button_height):
            color_factor = y / button_height
            b = int(button_color[0] * (1 - color_factor) + button_color2[0] * color_factor)
            g = int(button_color[1] * (1 - color_factor) + button_color2[1] * color_factor)
            r = int(button_color[2] * (1 - color_factor) + button_color2[2] * color_factor)
            cv2.line(background, 
                    (button_x, button_y + y + y_offset), 
                    (button_x + button_width, button_y + y + y_offset), 
                    (b, g, r), 1)
        
        # Draw play button border
        border_color = (255, 255, 255) if play_hover else (200, 200, 200)
        cv2.rectangle(background, 
                     (button_x, button_y + y_offset),
                     (button_x + button_width, button_y + button_height + y_offset),
                     border_color, 2)
        
        # Play button text
        play_text = "PLAY"
        text_size = cv2.getTextSize(play_text, font, 1, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2 + y_offset
        
        cv2.putText(background, play_text, (text_x + 2, text_y + 2), font, 1, (0, 50, 20), 2, cv2.LINE_AA)
        cv2.putText(background, play_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Exit button
        exit_button_width, exit_button_height = 150, 60
        exit_button_x = (self.width - exit_button_width) // 2
        exit_button_y = 610  # Increased from 560 to maintain spacing
        
        exit_hover = self.button_states['exit']['hover']
        exit_click = self.button_states['exit']['click']
        exit_color = (200, 50, 50) if exit_hover else (150, 30, 30)
        exit_color2 = (255, 80, 80) if exit_hover else (200, 50, 50)
        
        # Draw exit button with gradient
        y_offset = 3 if exit_click else 0
        for y in range(exit_button_height):
            color_factor = y / exit_button_height
            b = int(exit_color[0] * (1 - color_factor) + exit_color2[0] * color_factor)
            g = int(exit_color[1] * (1 - color_factor) + exit_color2[1] * color_factor)
            r = int(exit_color[2] * (1 - color_factor) + exit_color2[2] * color_factor)
            cv2.line(background, 
                    (exit_button_x, exit_button_y + y + y_offset), 
                    (exit_button_x + exit_button_width, exit_button_y + y + y_offset), 
                    (b, g, r), 1)
        
        # Draw exit button border
        border_color = (255, 255, 255) if exit_hover else (200, 200, 200)
        cv2.rectangle(background, 
                     (exit_button_x, exit_button_y + y_offset),
                     (exit_button_x + exit_button_width, exit_button_y + exit_button_height + y_offset),
                     border_color, 2)
        
        # Exit button text
        exit_text = "EXIT"
        text_size = cv2.getTextSize(exit_text, font, 0.8, 2)[0]
        text_x = exit_button_x + (exit_button_width - text_size[0]) // 2
        text_y = exit_button_y + (exit_button_height + text_size[1]) // 2 + y_offset
        
        cv2.putText(background, exit_text, (text_x + 2, text_y + 2), font, 0.8, (50, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(background, exit_text, (text_x, text_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Instructions button
        instructions_text = "INSTRUCTIONS"
        instructions_text_size = cv2.getTextSize(instructions_text, font, 0.7, 2)[0]
        instructions_button_width = instructions_text_size[0] + 60  # Add padding
        instructions_button_height = 50
        
        # Threshold control button
        threshold_text = "THRESHOLD CONTROLS"
        threshold_text_size = cv2.getTextSize(threshold_text, font, 0.7, 2)[0]
        threshold_button_width = threshold_text_size[0] + 60  # Add padding
        threshold_button_height = 50
        
        # Download button
        download_text = "DOWNLOAD REPORT"
        download_text_size = cv2.getTextSize(download_text, font, 0.7, 2)[0]
        download_button_width = download_text_size[0] + 60  # Add padding
        download_button_height = 50
        
        # Calculate total width needed for all three buttons plus gaps
        total_width = instructions_button_width + threshold_button_width + download_button_width + 80  # 40px gap between each button
        start_x = (self.width - total_width) // 2  # Center the entire button group
        
        # Position all buttons in the same row
        instructions_button_x = start_x
        instructions_button_y = 700
        
        threshold_button_x = start_x + instructions_button_width + 40  # 40px gap
        threshold_button_y = 700  # Same y as instructions button
        
        download_button_x = start_x + instructions_button_width + threshold_button_width + 80  # 80px from start (40px gap after each button)
        download_button_y = 700  # Same y as other buttons
        
        # Draw instructions button
        instructions_hover = self.button_states['instructions']['hover']
        instructions_click = self.button_states['instructions']['click']
        instructions_color = (0, 100, 200) if instructions_hover else (0, 80, 150)
        instructions_color2 = (0, 150, 255) if instructions_hover else (0, 120, 200)
        
        # Draw instructions button with gradient
        y_offset = 3 if instructions_click else 0
        for y in range(instructions_button_height):
            color_factor = y / instructions_button_height
            b = int(instructions_color[0] * (1 - color_factor) + instructions_color2[0] * color_factor)
            g = int(instructions_color[1] * (1 - color_factor) + instructions_color2[1] * color_factor)
            r = int(instructions_color[2] * (1 - color_factor) + instructions_color2[2] * color_factor)
            cv2.line(background, 
                    (instructions_button_x, instructions_button_y + y + y_offset), 
                    (instructions_button_x + instructions_button_width, instructions_button_y + y + y_offset), 
                    (b, g, r), 1)
        
        # Draw instructions button border
        border_color = (255, 255, 255) if instructions_hover else (200, 200, 200)
        cv2.rectangle(background, 
                     (instructions_button_x, instructions_button_y + y_offset),
                     (instructions_button_x + instructions_button_width, instructions_button_y + instructions_button_height + y_offset),
                     border_color, 2)
        
        # Instructions button text
        text_x = instructions_button_x + (instructions_button_width - instructions_text_size[0]) // 2
        text_y = instructions_button_y + (instructions_button_height + instructions_text_size[1]) // 2 + y_offset
        
        cv2.putText(background, instructions_text, (text_x + 2, text_y + 2), font, 0.7, (0, 20, 50), 2, cv2.LINE_AA)
        cv2.putText(background, instructions_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw threshold button
        threshold_hover = self.button_states['threshold']['hover']
        threshold_click = self.button_states['threshold']['click']
        threshold_color = (100, 100, 100) if threshold_hover else (80, 80, 80)
        threshold_color2 = (150, 150, 150) if threshold_hover else (120, 120, 120)
        
        # Draw threshold button with gradient
        y_offset = 3 if threshold_click else 0
        for y in range(threshold_button_height):
            color_factor = y / threshold_button_height
            b = int(threshold_color[0] * (1 - color_factor) + threshold_color2[0] * color_factor)
            g = int(threshold_color[1] * (1 - color_factor) + threshold_color2[1] * color_factor)
            r = int(threshold_color[2] * (1 - color_factor) + threshold_color2[2] * color_factor)
            cv2.line(background, 
                    (threshold_button_x, threshold_button_y + y + y_offset), 
                    (threshold_button_x + threshold_button_width, threshold_button_y + y + y_offset), 
                    (b, g, r), 1)
        
        # Draw threshold button border
        border_color = (255, 255, 255) if threshold_hover else (200, 200, 200)
        cv2.rectangle(background, 
                     (threshold_button_x, threshold_button_y + y_offset),
                     (threshold_button_x + threshold_button_width, threshold_button_y + threshold_button_height + y_offset),
                     border_color, 2)
        
        # Threshold button text
        text_x = threshold_button_x + (threshold_button_width - threshold_text_size[0]) // 2
        text_y = threshold_button_y + (threshold_button_height + threshold_text_size[1]) // 2 + y_offset
        
        cv2.putText(background, threshold_text, (text_x + 2, text_y + 2), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(background, threshold_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw download button
        download_hover = self.button_states['download']['hover']
        download_click = self.button_states['download']['click']
        download_color = (0, 150, 0) if download_hover else (0, 120, 0)
        download_color2 = (0, 200, 0) if download_hover else (0, 150, 0)
        
        # Draw download button with gradient
        y_offset = 3 if download_click else 0
        for y in range(download_button_height):
            color_factor = y / download_button_height
            b = int(download_color[0] * (1 - color_factor) + download_color2[0] * color_factor)
            g = int(download_color[1] * (1 - color_factor) + download_color2[1] * color_factor)
            r = int(download_color[2] * (1 - color_factor) + download_color2[2] * color_factor)
            cv2.line(background, 
                    (download_button_x, download_button_y + y + y_offset), 
                    (download_button_x + download_button_width, download_button_y + y + y_offset), 
                    (b, g, r), 1)
        
        # Draw download button border
        border_color = (255, 255, 255) if download_hover else (200, 200, 200)
        cv2.rectangle(background, 
                     (download_button_x, download_button_y + y_offset),
                     (download_button_x + download_button_width, download_button_y + download_button_height + y_offset),
                     border_color, 2)
        
        # Download button text
        text_x = download_button_x + (download_button_width - download_text_size[0]) // 2
        text_y = download_button_y + (download_button_height + download_text_size[1]) // 2 + y_offset
        
        cv2.putText(background, download_text, (text_x + 2, text_y + 2), font, 0.7, (0, 50, 0), 2, cv2.LINE_AA)
        cv2.putText(background, download_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show instructions if button was clicked
        if self.show_instructions:
            # Create semi-transparent overlay
            overlay = background.copy()
            cv2.rectangle(overlay, (self.width//8, self.height//8), 
                         (7 * self.width // 8, 7 * self.height // 8), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, background, 0.2, 0, background)
            
            # Add instructions title
            title = "INSTRUCTIONS"
            title_size = cv2.getTextSize(title, font, 1.2, 2)[0]
            title_x = (self.width - title_size[0]) // 2
            cv2.putText(background, title, (title_x, self.height//4 + 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Add instructions text
            y_pos = self.height//4 + 80
            for line in self.instructions_text:
                text_size = cv2.getTextSize(line, font, 0.7, 1)[0]
                text_x = (self.width - text_size[0]) // 2
                cv2.putText(background, line, (text_x, y_pos), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 30
        
        return background

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store original for visualization
        self.processing_images['original'] = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.processing_images['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply Gaussian blur
        blur_kernel = self.threshold_settings['blur_kernel']
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 3:
            blur_kernel = 3
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        self.processing_images['blurred'] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        # Apply binary thresholding
        _, binary_thresh = cv2.threshold(blurred, self.threshold_settings['binary'], 255, cv2.THRESH_BINARY_INV)
        self.processing_images['binary'] = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
        
        # Apply adaptive thresholding
        adaptive_size = self.threshold_settings['adaptive']
        if adaptive_size % 2 == 0:
            adaptive_size += 1
        if adaptive_size < 3:
            adaptive_size = 3
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_size,
            2
        )
        self.processing_images['adaptive'] = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
        
        # Apply Canny edge detection
        canny_edges = cv2.Canny(
            blurred,
            self.threshold_settings['canny_low'],
            self.threshold_settings['canny_high']
        )
        self.processing_images['canny'] = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        
        # Apply morphological operations
        morph_kernel = np.ones((self.threshold_settings['morph_kernel'], self.threshold_settings['morph_kernel']), np.uint8)
        morph = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, morph_kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, morph_kernel)
        self.processing_images['morphology'] = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        annotated_frame = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        self.processing_images['landmarks'] = annotated_frame
        
        return results

    def recognize_gesture(self, hand_landmarks):
        if not hand_landmarks:
            return None
        
        # Get fingertip landmarks
        landmarks = hand_landmarks.landmark
        
        # Check if fingers are extended
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip.x < thumb_ip.x  # For right hand
        
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_extended = index_tip.y < index_pip.y
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_extended = middle_tip.y < middle_pip.y
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_extended = ring_tip.y < ring_pip.y
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        # Determine gesture
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return Gesture.ROCK
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return Gesture.PAPER
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.SCISSORS
        elif thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return Gesture.LIZARD
        elif index_extended and middle_extended and ring_extended and not pinky_extended:
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            separation = abs(index_mcp.x - middle_mcp.x)
            if separation > 0.04:
                return Gesture.SPOCK
        
        return None

    def create_threshold_controls(self):
        """Create threshold control window with trackbars"""
        try:
            # Check if window exists before destroying
            if cv2.getWindowProperty(self.threshold_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.threshold_window_name)
        except:
            pass  # Window doesn't exist, which is fine
        
        # Create new window
        cv2.namedWindow(self.threshold_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.threshold_window_name, 400, 300)
        
        # Create trackbars for each threshold parameter
        cv2.createTrackbar('Binary Threshold', self.threshold_window_name, 
                          self.threshold_settings['binary'], 255, self.update_threshold)
        cv2.createTrackbar('Adaptive Block Size', self.threshold_window_name, 
                          self.threshold_settings['adaptive'], 99, self.update_threshold)
        cv2.createTrackbar('Canny Low', self.threshold_window_name, 
                          self.threshold_settings['canny_low'], 255, self.update_threshold)
        cv2.createTrackbar('Canny High', self.threshold_window_name, 
                          self.threshold_settings['canny_high'], 255, self.update_threshold)
        cv2.createTrackbar('Morph Kernel', self.threshold_window_name, 
                          self.threshold_settings['morph_kernel'], 15, self.update_threshold)
        cv2.createTrackbar('Blur Kernel', self.threshold_window_name, 
                          self.threshold_settings['blur_kernel'], 15, self.update_threshold)

    def update_threshold(self, x):
        """Update threshold settings based on trackbar values"""
        try:
            self.threshold_settings['binary'] = cv2.getTrackbarPos('Binary Threshold', self.threshold_window_name)
            
            # Ensure adaptive block size is odd and greater than 1
            adaptive_size = cv2.getTrackbarPos('Adaptive Block Size', self.threshold_window_name)
            if adaptive_size % 2 == 0:
                adaptive_size += 1
            if adaptive_size < 3:
                adaptive_size = 3
            self.threshold_settings['adaptive'] = adaptive_size
            
            self.threshold_settings['canny_low'] = cv2.getTrackbarPos('Canny Low', self.threshold_window_name)
            self.threshold_settings['canny_high'] = cv2.getTrackbarPos('Canny High', self.threshold_window_name)
            
            # Ensure morph kernel is odd
            morph_size = cv2.getTrackbarPos('Morph Kernel', self.threshold_window_name)
            if morph_size % 2 == 0:
                morph_size += 1
            if morph_size < 3:
                morph_size = 3
            self.threshold_settings['morph_kernel'] = morph_size
            
            # Ensure blur kernel is odd
            blur_size = cv2.getTrackbarPos('Blur Kernel', self.threshold_window_name)
            if blur_size % 2 == 0:
                blur_size += 1
            if blur_size < 3:
                blur_size = 3
            self.threshold_settings['blur_kernel'] = blur_size
        except cv2.error:
            # Handle case where window or trackbars don't exist yet
            pass

    def run(self):
        """Run the game"""
        cap = self.recognizer.start_camera()
        prev_time = time.time()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = (x, y)
                
                # Calculate button positions based on text content
                play_text = "PLAY"
                exit_text = "EXIT"
                instructions_text = "INSTRUCTIONS"
                threshold_text = "THRESHOLD CONTROLS"
                download_text = "DOWNLOAD REPORT"
                
                # Get text sizes
                play_size = cv2.getTextSize(play_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                exit_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                instructions_size = cv2.getTextSize(instructions_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                threshold_size = cv2.getTextSize(threshold_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                download_size = cv2.getTextSize(download_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Calculate button dimensions with padding
                play_button_width = play_size[0] + 80
                play_button_height = 80
                exit_button_width = exit_size[0] + 60
                exit_button_height = 60
                instructions_button_width = instructions_size[0] + 60
                instructions_button_height = 50
                threshold_button_width = threshold_size[0] + 60
                threshold_button_height = 50
                download_button_width = download_size[0] + 60
                download_button_height = 50
                
                # Calculate button positions (centered)
                play_button_x = (self.width - play_button_width) // 2
                play_button_y = 500
                
                exit_button_x = (self.width - exit_button_width) // 2
                exit_button_y = 610
                
                # Calculate positions for all three buttons in the same row
                total_width = instructions_button_width + threshold_button_width + download_button_width + 80  # 40px gap between each button
                start_x = (self.width - total_width) // 2
                
                instructions_button_x = start_x
                instructions_button_y = 700
                
                threshold_button_x = start_x + instructions_button_width + 40
                threshold_button_y = 700  # Same y as instructions button
                
                download_button_x = start_x + instructions_button_width + threshold_button_width + 80
                download_button_y = 700  # Same y as other buttons
                
                # Update button hover states with precise boundaries
                self.button_states['play']['hover'] = (
                    play_button_x <= x <= play_button_x + play_button_width and
                    play_button_y <= y <= play_button_y + play_button_height
                )
                
                self.button_states['exit']['hover'] = (
                    exit_button_x <= x <= exit_button_x + exit_button_width and
                    exit_button_y <= y <= exit_button_y + exit_button_height
                )
                
                self.button_states['instructions']['hover'] = (
                    instructions_button_x <= x <= instructions_button_x + instructions_button_width and
                    instructions_button_y <= y <= instructions_button_y + instructions_button_height
                )
                
                self.button_states['threshold']['hover'] = (
                    threshold_button_x <= x <= threshold_button_x + threshold_button_width and
                    threshold_button_y <= y <= threshold_button_y + threshold_button_height
                )
                
                self.button_states['download']['hover'] = (
                    download_button_x <= x <= download_button_x + download_button_width and
                    download_button_y <= y <= download_button_y + download_button_height
                )
            
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Handle button clicks with improved detection
                if self.state == GameState.WELCOME:
                    # Check play button
                    if self.button_states['play']['hover']:
                        self.button_states['play']['click'] = True
                        self.button_states['play']['click_time'] = time.time()
                        self.state = GameState.WAITING
                    
                    # Check exit button
                    if self.button_states['exit']['hover']:
                        self.button_states['exit']['click'] = True
                        self.button_states['exit']['click_time'] = time.time()
                        return True  # Signal to exit
                    
                    # Check instructions button
                    if self.button_states['instructions']['hover']:
                        self.button_states['instructions']['click'] = True
                        self.button_states['instructions']['click_time'] = time.time()
                        self.show_instructions = not self.show_instructions
                    
                    # Check threshold button
                    if self.button_states['threshold']['hover']:
                        self.button_states['threshold']['click'] = True
                        self.button_states['threshold']['click_time'] = time.time()
                        self.show_threshold_controls = not self.show_threshold_controls
                        if self.show_threshold_controls:
                            self.create_threshold_controls()
                            # Force window to be visible
                            cv2.imshow(self.threshold_window_name, np.zeros((300, 400, 3), dtype=np.uint8))
                            cv2.waitKey(1)
                        else:
                            try:
                                if cv2.getWindowProperty(self.threshold_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                                    cv2.destroyWindow(self.threshold_window_name)
                            except:
                                pass  # Window doesn't exist, which is fine
                    
                    # Check download button
                    if self.button_states['download']['hover']:
                        self.button_states['download']['click'] = True
                        self.button_states['download']['click_time'] = time.time()
                        # Generate and save the report
                        report_path = self.generate_match_report()
                        print(f"Report generated and saved to: {report_path}")
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Reset all button click states
                for button in self.button_states.values():
                    button['click'] = False
            
            return False  # Continue running the game
        
        # Set mouse callback
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        try:
            while True:
                # Calculate delta time for animations
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                
                # Update particles
                self.update_particles(dt)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally for a more natural view
                frame = cv2.flip(frame, 1)
                
                # Process the frame for hand detection
                results = self.process_frame(frame)
                
                # Game state machine
                if self.state == GameState.WELCOME:
                    frame = self.draw_welcome_screen(frame)
                    
                    # Handle button clicks
                    if self.button_states['play']['click'] and current_time - self.button_states['play']['click_time'] < 0.2:
                        self.state = GameState.WAITING
                        self.button_states['play']['click'] = False
                    
                    if self.button_states['exit']['click'] and current_time - self.button_states['exit']['click_time'] < 0.2:
                        break
                
                elif self.state == GameState.WAITING:
                    # Show rules reminder
                    frame = self.draw_rules_reminder(frame)
                    
                    cv2.putText(frame, "Say 'Rock, Paper, Scissors, Lizard, Spock!' and press SPACE", 
                              (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                elif self.state == GameState.COUNTDOWN:
                    elapsed = current_time - self.countdown_start
                    count = 3 - int(elapsed)
                    
                    if count > 0:
                        cv2.putText(frame, str(count), (frame.shape[1]//2 - 20, frame.shape[0]//2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
                    else:
                        # Capture player's gesture
                        self.player_choice = self.recognize_gesture(results.multi_hand_landmarks[0]) if results.multi_hand_landmarks else None
                        
                        # Generate computer's choice
                        self.computer_choice = random.choice(list(Gesture))
                        
                        # Determine the winner
                        self.result = self.determine_winner(self.player_choice, self.computer_choice)
                        
                        # Switch to result state
                        self.state = GameState.RESULT
                
                elif self.state == GameState.RESULT:
                    # Draw the game UI with results
                    frame = self.draw_game_ui(frame, self.player_choice, self.computer_choice, self.result)
                    
                    cv2.putText(frame, "Press SPACE to play again", (10, frame.shape[0] - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow(self.window_name, frame)
                
                # Display processing steps
                y_offset = 0
                for name, img in self.processing_images.items():
                    display_img = cv2.resize(img, (320, 240))
                    cv2.putText(display_img, name, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(f'Processing: {name}', display_img)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == 32:  # SPACE key
                    if self.state == GameState.WAITING:
                        self.state = GameState.COUNTDOWN
                        self.countdown_start = current_time
                    elif self.state == GameState.RESULT:
                        self.state = GameState.WELCOME
                        self.player_choice = None
                        self.computer_choice = None
                        self.result = None
        
        finally:
            self.recognizer.release_camera()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    game = RockPaperScissorsLizardSpockGame()
    game.run() 