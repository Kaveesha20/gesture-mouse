import cv2
import random
import pyautogui
import torch
import torch.nn as nn
import numpy as np
import pickle
import time
import ctypes
from pynput.mouse import Button, Controller
import mediapipe as mp

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# --------------------------
# Load model
# --------------------------
class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )
    def forward(self, x):
        return self.net(x)

model = GestureNet()
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# --------------------------
# Setup
# --------------------------
mouse = Controller()
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils

GESTURE_NAMES = {
    '0': 'Move Mouse',
    '1': 'Left Click',
    '2': 'Right Click',
    '3': 'Double Click',
    '4': 'Screenshot',
    '5': 'Scroll Up',    
    '6': 'Scroll Down', 
}

last_action_time = 0
COOLDOWN = 0.5

# Smoothing                        ← add from here
import collections
smooth_x = collections.deque(maxlen=5)
smooth_y = collections.deque(maxlen=5)
dragging = False  

# --------------------------
# Predict gesture
# --------------------------
def predict_gesture(landmark_list):
    x = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        confidence = torch.softmax(out, dim=1).max().item()
        pred = torch.argmax(out, dim=1).item()
        label = str(le.inverse_transform([pred])[0])  # add str()
    return label, confidence

# --------------------------
# Perform action
# --------------------------
def perform_action(gesture, processed):
    global last_action_time
    now = time.time()

    if gesture == '0':
        if processed.multi_hand_landmarks:
            lm = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            x = int(lm.x * screen_width)
            y = int(lm.y * screen_height)
            smooth_x.append(x)
            smooth_y.append(y)
            avg_x = int(sum(smooth_x) / len(smooth_x))
            avg_y = int(sum(smooth_y) / len(smooth_y))
            ctypes.windll.user32.SetCursorPos(avg_x, avg_y)
            # mouse.position = (x, y)  # pynput fix

        return

    if now - last_action_time < COOLDOWN:
        return

    if gesture == '1':
        mouse.click(Button.left)
    elif gesture == '2':
        mouse.click(Button.right)
    elif gesture == '3':
        pyautogui.doubleClick()
    elif gesture == '4':
        img = pyautogui.screenshot()
        img.save(f"screenshot_{random.randint(1,1000)}.png")
        print("Screenshot saved")

    last_action_time = now

# --------------------------
# Main loop
# --------------------------
def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Gesture Mouse Control", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)
        landmark_list = []
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

        gesture_label = None
        confidence = 0.0

        if len(landmark_list) == 42:
            gesture_label, confidence = predict_gesture(landmark_list)

            if confidence > 0.85:
                perform_action(gesture_label, processed)

        # --- UI overlay ---
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        if gesture_label is not None:
            name = GESTURE_NAMES.get(gesture_label, gesture_label)
            color = (0, 255, 0) if confidence > 0.85 else (0, 165, 255)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Mouse Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()