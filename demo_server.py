import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyautogui
import time
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io
import random
from pynput.mouse import Button, Controller
import ctypes
import collections

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

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
model.load_state_dict(torch.load('gesture_model.pth', weights_only=True))
model.eval()

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Mouse setup
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Smoothing
smooth_x = collections.deque(maxlen=5)
smooth_y = collections.deque(maxlen=5)

# --------------------------
# MediaPipe
# --------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
draw = mp.solutions.drawing_utils

GESTURE_NAMES = {'0': 'Move Mouse', '1': 'Left Click', '2': 'Right Click',
                 '3': 'Double Click', '4': 'Screenshot'}

connected_clients = set()
last_action_time = 0
COOLDOWN = 0.5

async def register(websocket):
    connected_clients.add(websocket)
    print(f"Browser connected. Total: {len(connected_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)

async def broadcast(message):
    if connected_clients:
        await asyncio.gather(*[c.send(message) for c in connected_clients], return_exceptions=True)

# ====================== PREDICT & PERFORM ACTION ======================
def predict_gesture(landmark_list):
    x = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        confidence = torch.softmax(out, dim=1).max().item()
        pred = torch.argmax(out, dim=1).item()
        label = str(le.inverse_transform([pred])[0])
    return label, confidence

def perform_action(gesture, processed, confidence):
    global last_action_time
    now = time.time()

    if gesture == '0':  # Move Mouse
        if processed.multi_hand_landmarks:
            lm = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            x = int(lm.x * screen_width)
            y = int(lm.y * screen_height)
            smooth_x.append(x)
            smooth_y.append(y)
            avg_x = int(sum(smooth_x) / len(smooth_x))
            avg_y = int(sum(smooth_y) / len(smooth_y))
            ctypes.windll.user32.SetCursorPos(avg_x, avg_y)
        return

    if confidence < 0.85 or (now - last_action_time) < COOLDOWN:
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

# ====================== TRAINING ======================
async def train_model(data_csv_base64, epochs=30, lr=0.001, test_split=0.2):
    try:
        csv_bytes = base64.b64decode(data_csv_base64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values

        global le
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        net = GestureNet()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        await broadcast(json.dumps({"type": "train_status", "status": "starting", "message": "Training started..."}))

        for epoch in range(epochs):
            net.train()
            total_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            progress = int(((epoch + 1) / epochs) * 100)
            await broadcast(json.dumps({
                "type": "train_progress",
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "progress": progress,
                "loss": round(total_loss / len(train_loader), 4)
            }))

        # Save
        torch.save(net.state_dict(), 'gesture_model.pth')
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)

        global model
        model = net
        model.eval()

        await broadcast(json.dumps({
            "type": "train_status",
            "status": "completed",
            "message": "✅ Training completed! Model updated successfully."
        }))

    except Exception as e:
        await broadcast(json.dumps({"type": "train_status", "status": "error", "message": str(e)}))

# ====================== CAMERA LOOP ======================
async def camera_loop():
    global last_action_time
    cap = cv2.VideoCapture(0)

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
        action = None

        if len(landmark_list) == 42:
            gesture_label, confidence = predict_gesture(landmark_list)
            perform_action(gesture_label, processed, confidence)

            if gesture_label == '0':
                action = "move"
            elif confidence > 0.85:
                if gesture_label == '1': action = "left_click"
                elif gesture_label == '2': action = "right_click"
                elif gesture_label == '3': action = "double_click"
                elif gesture_label == '4': action = "screenshot"

        # Encode frame for UI
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "gesture": gesture_label,
            "gesture_name": GESTURE_NAMES.get(gesture_label, "No gesture"),
            "confidence": round(confidence * 100, 1),
            "cursor_x": round(cursor_x, 4),
            "cursor_y": round(cursor_y, 4),
            "action": action,
            "landmarks": landmark_list if len(landmark_list) == 42 else [],
            "frame": frame_base64
        }

        await broadcast(json.dumps(payload))
        await asyncio.sleep(0.025)  # ~40 FPS

    cap.release()

# ====================== WEBSOCKET ======================
async def handler(websocket):
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "train":
                await train_model(
                    data["csv_base64"],
                    epochs=data.get("epochs", 30),
                    lr=data.get("lr", 0.001),
                    test_split=data.get("test_split", 0.2)
                )
    except:
        pass

async def main():
    print("Gesture Lab Server Running → ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await camera_loop()

if __name__ == "__main__":
    asyncio.run(main())