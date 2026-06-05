import asyncio
import websockets
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyautogui
import time

# --------------------------
# Model
# --------------------------
class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

model = GestureNet()
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def predict_gesture(landmark_list):
    x = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        confidence = torch.softmax(out, dim=1).max().item()
        pred = torch.argmax(out, dim=1).item()
    label = str(le.inverse_transform([pred])[0])
    return label, confidence

# --------------------------
# MediaPipe
# --------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

GESTURE_NAMES = {
    '0': 'Move Mouse',
    '1': 'Left Click',
    '2': 'Right Click',
    '3': 'Double Click',
    '4': 'Screenshot'
}

# --------------------------
# Connected clients
# --------------------------
connected_clients = set()

async def register(websocket):
    connected_clients.add(websocket)
    print(f"Browser connected. Total: {len(connected_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"Browser disconnected. Total: {len(connected_clients)}")

async def broadcast(message):
    if connected_clients:
        await asyncio.gather(*[c.send(message) for c in connected_clients], return_exceptions=True)

# --------------------------
# Camera loop
# --------------------------
last_action_time = 0
COOLDOWN = 0.6

async def camera_loop():
    global last_action_time
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Gesture Camera", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmark_list = []
        cursor_x, cursor_y = 0.5, 0.5

        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
            tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            cursor_x, cursor_y = tip.x, tip.y

        gesture_label = None
        confidence = 0.0

        if len(landmark_list) == 42:
            gesture_label, confidence = predict_gesture(landmark_list)
            now = time.time()

            payload = {
                "gesture": gesture_label,
                "gesture_name": GESTURE_NAMES.get(gesture_label, gesture_label),
                "confidence": round(confidence * 100, 1),
                "cursor_x": round(cursor_x, 4),
                "cursor_y": round(cursor_y, 4),
                "action": None
            }

            if gesture_label == '0':
                payload["action"] = "move"
                await broadcast(json.dumps(payload))

            elif confidence > 0.85 and (now - last_action_time) > COOLDOWN:
                if gesture_label == '1':
                    payload["action"] = "left_click"
                elif gesture_label == '2':
                    payload["action"] = "right_click"
                elif gesture_label == '3':
                    payload["action"] = "double_click"
                elif gesture_label == '4':
                    payload["action"] = "screenshot"
                last_action_time = now
                await broadcast(json.dumps(payload))

        # Camera overlay
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
        if gesture_label:
            name = GESTURE_NAMES.get(gesture_label, gesture_label)
            color = (0, 255, 150) if confidence > 0.85 else (0, 165, 255)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Gesture Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    print("Open demo_ui.html in your browser, then show your hand!")
    async with websockets.serve(register, "localhost", 8765):
        await camera_loop()

if __name__ == "__main__":
    asyncio.run(main())