# pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

# --- Configuration & Constants ---
WIDTH, HEIGHT = 1280, 720
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 140
BALL_RADIUS = 15
WIN_SCORE = 11

# Colors (BGR)
COLOR_TABLE = (160, 70, 20)      # Ping pong table blue
COLOR_LINE = (255, 255, 255)     # White lines
COLOR_NET = (150, 150, 150)      # Net gray
COLOR_BALL = (0, 165, 255)       # Orange ball
COLOR_TRAIL = (0, 200, 255)      # Trail color
COLOR_SMASH_TRAIL = (0, 0, 255)  # Red trail for smash
COLOR_P1 = (80, 80, 255)         # P1 Paddle (Reddish)
COLOR_P2 = (255, 100, 50)        # P2 Paddle (Blueish)
COLOR_TEXT = (255, 255, 255)

# Game Rules
INITIAL_BALL_SPEED = 12
SPEED_INCREMENT = 1.0
MAX_BALL_SPEED = 35
SMASH_MULTIPLIER = 1.5
SMASH_VELOCITY_THRESHOLD = 800  # pixels per second of paddle movement

def create_table_bg(width, height):
    """Draws the static background of the ping pong table to optimize performance."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = COLOR_TABLE
    
    # Borders
    cv2.rectangle(bg, (10, 10), (width - 10, height - 10), COLOR_LINE, 10)
    
    # Center division
    cv2.line(bg, (width // 2, 10), (width // 2, height - 10), COLOR_LINE, 5)
    
    # Draw Net
    for y in range(10, height - 10, 25):
        cv2.line(bg, (width // 2, y), (width // 2, y + 12), COLOR_NET, 8)
        
    return bg

class AirPongGame:
    def __init__(self):
        # Paddle state
        self.p1_y = HEIGHT // 2
        self.p2_y = HEIGHT // 2
        self.p1_prev_y = self.p1_y
        self.p2_prev_y = self.p2_y
        self.p1_hit_glow = 0
        self.p2_hit_glow = 0
        
        # Ball state
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_vx = INITIAL_BALL_SPEED if np.random.rand() > 0.5 else -INITIAL_BALL_SPEED
        self.ball_vy = np.random.choice([-INITIAL_BALL_SPEED // 2, INITIAL_BALL_SPEED // 2])
        self.current_speed = INITIAL_BALL_SPEED
        self.ball_trail = deque(maxlen=15)
        self.is_smash = False
        
        # Game Match State
        self.score1 = 0
        self.score2 = 0
        self.state = "PLAYING" # PLAYING, P1_WINS, P2_WINS
        
        self.last_time = time.time()
        
    def reset_ball(self, scored_player):
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.current_speed = INITIAL_BALL_SPEED
        self.is_smash = False
        self.ball_trail.clear()
        
        # Serve towards the player who lost the point
        direction = 1 if scored_player == 1 else -1
        self.ball_vx = INITIAL_BALL_SPEED * direction
        self.ball_vy = np.random.choice([-INITIAL_BALL_SPEED // 2, INITIAL_BALL_SPEED // 2])
        
    def check_victory(self):
        if self.score1 >= WIN_SCORE and self.score1 - self.score2 >= 2:
            self.state = "P1_WINS"
        elif self.score2 >= WIN_SCORE and self.score2 - self.score1 >= 2:
            self.state = "P2_WINS"

    def update_physics(self, dt):
        if self.state != "PLAYING":
            return
            
        # Update ball position
        self.ball_x += int(self.ball_vx)
        self.ball_y += int(self.ball_vy)
        
        self.ball_trail.appendleft((self.ball_x, self.ball_y))
        
        # Wall bounces (Top and Bottom)
        if self.ball_y - BALL_RADIUS <= 10:
            self.ball_y = 10 + BALL_RADIUS
            self.ball_vy = abs(self.ball_vy) # Bounce down
            # Optional: Add bounce sound placeholder here
            
        elif self.ball_y + BALL_RADIUS >= HEIGHT - 10:
            self.ball_y = HEIGHT - 10 - BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy) # Bounce up
            
        # Paddle collisions
        # P1 Paddle (Left)
        p1_x_front = 50 + PADDLE_WIDTH
        p1_top = self.p1_y - PADDLE_HEIGHT // 2
        p1_bottom = self.p1_y + PADDLE_HEIGHT // 2
        
        if self.ball_vx < 0 and self.ball_x - BALL_RADIUS <= p1_x_front and self.ball_x + BALL_RADIUS >= 50:
            if p1_top <= self.ball_y <= p1_bottom:
                self.ball_x = p1_x_front + BALL_RADIUS
                self.handle_paddle_hit(1, dt)
                
        # P2 Paddle (Right)
        p2_x_front = WIDTH - 50 - PADDLE_WIDTH
        p2_top = self.p2_y - PADDLE_HEIGHT // 2
        p2_bottom = self.p2_y + PADDLE_HEIGHT // 2
        
        if self.ball_vx > 0 and self.ball_x + BALL_RADIUS >= p2_x_front and self.ball_x - BALL_RADIUS <= WIDTH - 50:
            if p2_top <= self.ball_y <= p2_bottom:
                self.ball_x = p2_x_front - BALL_RADIUS
                self.handle_paddle_hit(2, dt)

        # Scoring Logic (crossing left/right boundaries)
        if self.ball_x < 0:
            self.score2 += 1
            self.check_victory()
            if self.state == "PLAYING":
                self.reset_ball(scored_player=2)
                
        elif self.ball_x > WIDTH:
            self.score1 += 1
            self.check_victory()
            if self.state == "PLAYING":
                self.reset_ball(scored_player=1)
                
    def handle_paddle_hit(self, player_id, dt):
        """Calculates ball trajectory, speed increase, and smash detection after a paddle hit."""
        paddle_y = self.p1_y if player_id == 1 else self.p2_y
        prev_paddle_y = self.p1_prev_y if player_id == 1 else self.p2_prev_y
        
        # Trigger Glow
        if player_id == 1: self.p1_hit_glow = 10
        else: self.p2_hit_glow = 10

        # Calculate paddle velocity
        paddle_vel = (paddle_y - prev_paddle_y) / (dt + 1e-5)
        
        # Determine if hit is a smash
        self.is_smash = abs(paddle_vel) > SMASH_VELOCITY_THRESHOLD
        
        # Increase ball speed
        if self.is_smash:
            self.current_speed = min(self.current_speed * SMASH_MULTIPLIER, MAX_BALL_SPEED)
        else:
            self.current_speed = min(self.current_speed + SPEED_INCREMENT, MAX_BALL_SPEED)
            
        # Change angle based on where ball hits paddle (relative intersect)
        relative_intersect_y = paddle_y - self.ball_y
        normalized_intersect = relative_intersect_y / (PADDLE_HEIGHT / 2) # Range: -1 to 1
        bounce_angle = normalized_intersect * (math.pi / 3) # Max 60 degree bounce
        
        # If moving from paddle velocity, add spin/angle variation
        bounce_angle -= np.clip((paddle_vel * 0.001), -0.5, 0.5)

        direction = 1 if player_id == 1 else -1
        self.ball_vx = direction * self.current_speed * math.cos(bounce_angle)
        self.ball_vy = -self.current_speed * math.sin(bounce_angle) # Negative because y grows downward
        
    def render(self, frame, bg):
        # Blend background onto frame
        # Adding a darkened alpha layer to make game visible against webcam
        alpha = 0.8
        cv2.addWeighted(bg, alpha, frame, 1 - alpha, 0, frame)

        # Draw HUD Scores
        cv2.putText(frame, str(self.score1), (WIDTH // 4, 100), cv2.FONT_HERSHEY_DUPLEX, 4, COLOR_TEXT, 5)
        cv2.putText(frame, str(self.score2), (3 * WIDTH // 4, 100), cv2.FONT_HERSHEY_DUPLEX, 4, COLOR_TEXT, 5)
        
        # Draw Ball Trail
        for i, pos in enumerate(self.ball_trail):
            radius = max(2, BALL_RADIUS - i)
            trail_col = COLOR_SMASH_TRAIL if self.is_smash else COLOR_TRAIL
            cv2.circle(frame, pos, radius, trail_col, -1)

        # Draw Ball
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS, COLOR_BALL, -1)

        # Function to draw paddle with optional glow
        def draw_paddle(x, y, color, glow_strength):
            # Glow Effect
            if glow_strength > 0:
                glow_col = (min(255, color[0] + 100), min(255, color[1] + 100), min(255, color[2] + 100))
                cv2.rectangle(frame, (x - 8, y - PADDLE_HEIGHT//2 - 8), 
                              (x + PADDLE_WIDTH + 8, y + PADDLE_HEIGHT//2 + 8), glow_col, -1)
            # Paddle Body
            cv2.rectangle(frame, (x, y - PADDLE_HEIGHT//2), 
                          (x + PADDLE_WIDTH, y + PADDLE_HEIGHT//2), color, -1)
            
        # Draw Paddles
        draw_paddle(50, self.p1_y, COLOR_P1, self.p1_hit_glow)
        draw_paddle(WIDTH - 50 - PADDLE_WIDTH, self.p2_y, COLOR_P2, self.p2_hit_glow)
        
        # Degrade Glow
        self.p1_hit_glow = max(0, self.p1_hit_glow - 1)
        self.p2_hit_glow = max(0, self.p2_hit_glow - 1)
        
        # Match point / Victory Screen
        if self.state in ["P1_WINS", "P2_WINS"]:
            winner_text = "PLAYER 1 WINS!" if self.state == "P1_WINS" else "PLAYER 2 WINS!"
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_DUPLEX, 3, 5)[0]
            text_x = (WIDTH - text_size[0]) // 2
            text_y = HEIGHT // 2
            cv2.putText(frame, winner_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 6)
            
            restart_text = "Press 'R' to Restart or 'Q' to Quit"
            text_size2 = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, restart_text, ((WIDTH - text_size2[0]) // 2, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    game = AirPongGame()
    table_bg = create_table_bg(WIDTH, HEIGHT)

    last_time = time.time()
    fps_history = deque(maxlen=20)

    print("=== Air Pong Initialized ===")
    print("Use hands gestures to move paddles.")
    print("Press 'Q' to quit and 'R' to restart the match.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate Delta Time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        fps_history.append(1.0 / (dt + 1e-5))
        fps = sum(fps_history) / len(fps_history)

        # Flip frame horizontally for selfie-view display
        frame = cv2.flip(frame, 1)
        # Resize to match our logical game resolution
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Hand Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Store previous paddle y positions for physics (smash detection)
        game.p1_prev_y = game.p1_y
        game.p2_prev_y = game.p2_y

        # Process Hands
        if results.multi_hand_landmarks:
            detected_hands = []
            for hand_landmarks in results.multi_hand_landmarks:
                # We use the wrist (0) or palm (9 - middle finger mcp) as the tracking point
                # Middle finger MCP provides a more central, stable anchor than wrist sometimes.
                cx = int(hand_landmarks.landmark[9].x * WIDTH)
                cy = int(hand_landmarks.landmark[9].y * HEIGHT)
                detected_hands.append((cx, cy, hand_landmarks))

            # Sort hands by x-coordinate to easily assign left player and right player
            detected_hands.sort(key=lambda item: item[0])
            
            for cx, cy, landmarks in detected_hands:
                # Draw landmarks subtly 
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1)
                )

                # Assign to paddle
                if cx < WIDTH // 2:
                    game.p1_y = np.clip(cy, PADDLE_HEIGHT // 2, HEIGHT - PADDLE_HEIGHT // 2)
                else:
                    game.p2_y = np.clip(cy, PADDLE_HEIGHT // 2, HEIGHT - PADDLE_HEIGHT // 2)

        # Update Physics
        game.update_physics(dt)

        # Render Game state over the frame
        game.render(frame, table_bg)

        # Draw FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Air Pong", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Manual Restart
            game = AirPongGame()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
