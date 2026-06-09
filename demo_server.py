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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io
import os
import random
from pynput.mouse import Button, Controller
import ctypes
import collections

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

DATA_FILE = 'gesture_data.csv'

# Separate model files
DEFAULT_MODEL_FILE = 'gesture_model_default.pth'
DEFAULT_ENCODER_FILE = 'label_encoder_default.pkl'
CUSTOM_MODEL_FILE = 'gesture_model_custom.pth'
CUSTOM_ENCODER_FILE = 'label_encoder_custom.pkl'

# Current mode
current_mode = 'default'

class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 7)
        )
    def forward(self, x):
        return self.net(x)

model = GestureNet()
le = LabelEncoder()
training_lock = asyncio.Lock()

def load_model():
    global model, le, current_mode, model_ready
    
    print(f"\n[LOAD] Loading model in '{current_mode.upper()}' mode...")
    
    if current_mode == 'customize':
        # Try custom model first
        if os.path.exists(CUSTOM_MODEL_FILE) and os.path.exists(CUSTOM_ENCODER_FILE):
            model.load_state_dict(torch.load(CUSTOM_MODEL_FILE, weights_only=True))
            with open(CUSTOM_ENCODER_FILE, 'rb') as f:
                le = pickle.load(f)
            model.eval()
            print("[LOAD] ✓ CUSTOM model loaded")
            return True
        elif os.path.exists(DEFAULT_MODEL_FILE) and os.path.exists(DEFAULT_ENCODER_FILE):
            model.load_state_dict(torch.load(DEFAULT_MODEL_FILE, weights_only=True))
            with open(DEFAULT_ENCODER_FILE, 'rb') as f:
                le = pickle.load(f)
            model.eval()
            print("[LOAD] ✓ DEFAULT model loaded (fallback)")
            return True
    else:
        # Default mode - only use default model
        if os.path.exists(DEFAULT_MODEL_FILE) and os.path.exists(DEFAULT_ENCODER_FILE):
            model.load_state_dict(torch.load(DEFAULT_MODEL_FILE, weights_only=True))
            with open(DEFAULT_ENCODER_FILE, 'rb') as f:
                le = pickle.load(f)
            model.eval()
            print("[LOAD] ✓ DEFAULT model loaded (fixed)")
            return True
    
    print("[LOAD] ✗ No model found!")
    le.fit(['0', '1', '2', '3', '4', '5', '6'])
    model.eval()
    return False

# Mouse setup
mouse = Controller()
screen_width, screen_height = pyautogui.size()
smooth_x = collections.deque(maxlen=5)
smooth_y = collections.deque(maxlen=5)

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1,
)
draw = mp.solutions.drawing_utils

GESTURE_NAMES = {
    '0': 'Move Mouse', '1': 'Left Click', '2': 'Right Click',
    '3': 'Double Click', '4': 'Screenshot',
    '5': 'Scroll Up', '6': 'Scroll Down', 
}

connected_clients = set()
last_action_time = 0
COOLDOWN = 2.0
model_ready = load_model()

def get_dataset_info():
    if not os.path.exists(DATA_FILE):
        return {"total": 0, "labels": {}}
    df = pd.read_csv(DATA_FILE)
    labels = df['label'].value_counts().sort_index().to_dict()
    labels = {str(k): int(v) for k, v in labels.items()}
    return {"total": len(df), "labels": labels}

def merge_csv_data(csv_base64):
    csv_bytes = base64.b64decode(csv_base64)
    new_df = pd.read_csv(io.BytesIO(csv_bytes))
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(DATA_FILE, index=False)
    return len(new_df), len(combined)

async def register(websocket):
    connected_clients.add(websocket)
    print(f"[CONNECT] Browser connected. Total: {len(connected_clients)}")
    info = get_dataset_info()
    await websocket.send(json.dumps({
        "type": "dataset_info",
        "total": info["total"],
        "labels": info["labels"],
        "model_ready": model_ready,
        "current_mode": current_mode,
    }))

async def unregister(websocket):
    connected_clients.discard(websocket)
    print(f"[DISCONNECT] Browser disconnected. Total: {len(connected_clients)}")

async def broadcast(message):
    if connected_clients:
        await asyncio.gather(*[c.send(message) for c in connected_clients], return_exceptions=True)

def predict_gesture(landmark_list):
    if not model_ready:
        return None, 0.0
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
        return "move"

    if confidence < 0.85 or (now - last_action_time) < COOLDOWN:
        return None

    action_map = {
        '1': ("left_click", lambda: mouse.click(Button.left)),
        '2': ("right_click", lambda: mouse.click(Button.right)),
        '3': ("double_click", pyautogui.doubleClick),
        '4': ("screenshot", lambda: pyautogui.screenshot().save(f"screenshot_{random.randint(1,1000)}.png")),
        '5': ("scroll_up", lambda: pyautogui.scroll(3)),
        '6': ("scroll_down", lambda: pyautogui.scroll(-3)),
    }
    
    if gesture in action_map:
        action, func = action_map[gesture]
        func()
        last_action_time = now
        return action
    return None

async def save_data(csv_base64):
    try:
        added, total = merge_csv_data(csv_base64)
        await broadcast(json.dumps({
            "type": "save_status",
            "status": "completed",
            "message": f"Saved {added} rows. Dataset: {total} rows.",
            "total": total,
        }))
        await broadcast(json.dumps({
            "type": "dataset_info",
            "total": total,
            "labels": get_dataset_info()["labels"],
            "model_ready": model_ready,
            "current_mode": current_mode,
        }))
    except Exception as e:
        await broadcast(json.dumps({"type": "save_status", "status": "error", "message": str(e)}))

async def train_model(csv_base64=None, epochs=30, lr=0.001, test_split=0.2):
    global model, le, model_ready, current_mode
    async with training_lock:
        try:
            if csv_base64:
                added, total = merge_csv_data(csv_base64)
                await broadcast(json.dumps({"type": "train_status", "status": "starting", "message": f"Merged {added} rows. Training CUSTOM model..."}))
            
            df = pd.read_csv(DATA_FILE)
            if len(df) < 10:
                raise ValueError("Need at least 10 samples to train.")
            
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df['label'].values
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
            
            train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
            test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32)
            
            net = GestureNet()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                net.train()
                total_loss = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(net(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                net.eval()
                correct = 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        preds = torch.argmax(net(xb), dim=1)
                        correct += (preds == yb).sum().item()
                val_acc = correct / len(test_ds) * 100
                
                if (epoch + 1) % max(1, epochs//10) == 0:
                    await broadcast(json.dumps({"type": "train_progress", "epoch": epoch+1, "total_epochs": epochs, "loss": round(total_loss/len(train_loader),4), "val_acc": round(val_acc,1)}))
            
            torch.save(net.state_dict(), CUSTOM_MODEL_FILE)
            with open(CUSTOM_ENCODER_FILE, 'wb') as f:
                pickle.dump(le, f)
            
            if current_mode == 'customize':
                model = net
                model.eval()
                model_ready = True
            
            await broadcast(json.dumps({"type": "train_status", "status": "completed", "message": f"Training complete! Custom model saved.", "val_acc": round(val_acc,1)}))
        except Exception as e:
            await broadcast(json.dumps({"type": "train_status", "status": "error", "message": str(e)}))

async def camera_loop():
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
        
        gesture_label, confidence, action = None, 0.0, None
        if len(landmark_list) == 42:
            gesture_label, confidence = predict_gesture(landmark_list)
            if gesture_label is not None:
                action = perform_action(gesture_label, processed, confidence)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        await broadcast(json.dumps({
            "type": "frame",
            "gesture": gesture_label,
            "confidence": round(confidence * 100, 1),
            "cursor_x": round(cursor_x, 4),
            "cursor_y": round(cursor_y, 4),
            "action": action,
            "landmarks": landmark_list if len(landmark_list) == 42 else [],
            "frame": base64.b64encode(buffer).decode('utf-8'),
            "model_ready": model_ready,
            "current_mode": current_mode,
        }))
        await asyncio.sleep(0.025)
    cap.release()

async def handler(websocket):
    global current_mode, model, le, model_ready
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "save_data":
                await save_data(data["csv_base64"])
            elif data.get("type") == "train":
                asyncio.create_task(train_model(data.get("csv_base64"), data.get("epochs", 30), data.get("lr", 0.001), data.get("test_split", 0.2)))
            elif data.get("type") == "get_dataset":
                await websocket.send(json.dumps({**get_dataset_info(), "type": "dataset_info", "model_ready": model_ready, "current_mode": current_mode}))
            elif data.get("type") == "set_mode":
                new_mode = data.get("mode", "default")
                if new_mode != current_mode:
                    current_mode = new_mode
                    print(f"\n[MODE] Switching to {current_mode.upper()} mode...")
                    model_ready = load_model()
                    await websocket.send(json.dumps({"type": "mode_changed", "mode": current_mode, "model_ready": model_ready}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(websocket)

async def main():
    print("\n" + "="*50)
    print("GESTURE LAB SERVER")
    print("="*50)
    print(f"Default Model: {DEFAULT_MODEL_FILE}")
    print(f"Custom Model: {CUSTOM_MODEL_FILE}")
    print(f"Current Mode: {current_mode.upper()}")
    print("\nMake sure you have created gesture_model_default.pth first!")
    print("Run: python create_default_model.py")
    print("="*50 + "\n")
    async with websockets.serve(handler, "localhost", 8765):
        await camera_loop()

if __name__ == "__main__":
    asyncio.run(main())