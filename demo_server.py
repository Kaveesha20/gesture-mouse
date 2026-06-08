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
MODEL_FILE = 'gesture_model.pth'
ENCODER_FILE = 'label_encoder.pkl'

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
le = LabelEncoder()
training_lock = asyncio.Lock()

def load_model():
    global model, le
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        with open(ENCODER_FILE, 'rb') as f:
            le = pickle.load(f)
        model.eval()
        print("Loaded trained model.")
        return True
    le.fit([0, 1, 2, 3, 4])
    model.eval()
    print("No trained model found — collect data and train from the UI.")
    return False

model_ready = load_model()

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
}

connected_clients = set()
last_action_time = 0
COOLDOWN = 2.0


def get_dataset_info():
    if not os.path.exists(DATA_FILE):
        return {"total": 0, "labels": {}, "path": DATA_FILE}

    df = pd.read_csv(DATA_FILE)
    labels = df['label'].value_counts().sort_index().to_dict()
    labels = {str(k): int(v) for k, v in labels.items()}
    return {"total": len(df), "labels": labels, "path": DATA_FILE}


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
    print(f"Browser connected. Total: {len(connected_clients)}")
    info = get_dataset_info()
    await websocket.send(json.dumps({
        "type": "dataset_info",
        "total": info["total"],
        "labels": info["labels"],
        "model_ready": model_ready,
    }))


async def unregister(websocket):
    connected_clients.discard(websocket)
    print(f"Browser disconnected. Total: {len(connected_clients)}")


async def broadcast(message):
    if connected_clients:
        await asyncio.gather(
            *[c.send(message) for c in connected_clients],
            return_exceptions=True,
        )


async def send_dataset_update():
    info = get_dataset_info()
    await broadcast(json.dumps({
        "type": "dataset_info",
        "total": info["total"],
        "labels": info["labels"],
        "model_ready": model_ready,
    }))


# ====================== PREDICT & PERFORM ACTION ======================
def predict_gesture(landmark_list):
    if not model_ready and not os.path.exists(MODEL_FILE):
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

    if gesture == '1':
        mouse.click(Button.left)
        action = "left_click"
    elif gesture == '2':
        mouse.click(Button.right)
        action = "right_click"
    elif gesture == '3':
        pyautogui.doubleClick()
        action = "double_click"
    elif gesture == '4':
        img = pyautogui.screenshot()
        img.save(f"screenshot_{random.randint(1, 1000)}.png")
        print("Screenshot saved")
        action = "screenshot"
    else:
        return None

    last_action_time = now
    return action


# ====================== DATA & TRAINING ======================
async def save_data(csv_base64):
    try:
        added, total = merge_csv_data(csv_base64)
        await broadcast(json.dumps({
            "type": "save_status",
            "status": "completed",
            "message": f"Saved {added} new rows. Dataset now has {total} rows.",
            "added": added,
            "total": total,
        }))
        await send_dataset_update()
    except Exception as e:
        await broadcast(json.dumps({
            "type": "save_status",
            "status": "error",
            "message": str(e),
        }))


async def train_model(csv_base64=None, epochs=30, lr=0.001, test_split=0.2):
    global model, le, model_ready

    async with training_lock:
        try:
            if csv_base64:
                added, total = merge_csv_data(csv_base64)
                await broadcast(json.dumps({
                    "type": "train_status",
                    "status": "starting",
                    "message": f"Merged {added} new rows into dataset ({total} total). Training...",
                }))
            elif not os.path.exists(DATA_FILE):
                raise FileNotFoundError("No dataset found. Collect samples in the UI first.")
            else:
                info = get_dataset_info()
                await broadcast(json.dumps({
                    "type": "train_status",
                    "status": "starting",
                    "message": f"Training on {info['total']} rows from {DATA_FILE}...",
                }))

            df = pd.read_csv(DATA_FILE)
            if len(df) < 10:
                raise ValueError("Need at least 10 samples to train.")

            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df['label'].values

            le = LabelEncoder()
            y = le.fit_transform(y)

            min_class_count = pd.Series(y).value_counts().min()
            stratify = y if min_class_count >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, stratify=stratify,
            )

            train_ds = TensorDataset(
                torch.tensor(X_train),
                torch.tensor(y_train, dtype=torch.long),
            )
            test_ds = TensorDataset(
                torch.tensor(X_test),
                torch.tensor(y_test, dtype=torch.long),
            )
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
                    out = net(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                net.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        out = net(xb)
                        preds = torch.argmax(out, dim=1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                val_acc = correct / total * 100 if total else 0

                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    await broadcast(json.dumps({
                        "type": "train_progress",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "loss": round(total_loss / len(train_loader), 4),
                        "val_acc": round(val_acc, 1),
                    }))

            torch.save(net.state_dict(), MODEL_FILE)
            with open(ENCODER_FILE, 'wb') as f:
                pickle.dump(le, f)

            model = net
            model.eval()
            model_ready = True

            await broadcast(json.dumps({
                "type": "train_status",
                "status": "completed",
                "message": f"Training completed. Model updated ({len(df)} samples, val acc {val_acc:.1f}%).",
                "val_acc": round(val_acc, 1),
                "total_samples": len(df),
            }))
            await send_dataset_update()

        except Exception as e:
            await broadcast(json.dumps({
                "type": "train_status",
                "status": "error",
                "message": str(e),
            }))


# ====================== CAMERA LOOP ======================
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

        gesture_label = None
        confidence = 0.0
        action = None

        if len(landmark_list) == 42:
            gesture_label, confidence = predict_gesture(landmark_list)
            if gesture_label is not None:
                action = perform_action(gesture_label, processed, confidence)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "type": "frame",
            "gesture": gesture_label,
            "gesture_name": GESTURE_NAMES.get(gesture_label, "No gesture"),
            "confidence": round(confidence * 100, 1),
            "cursor_x": round(cursor_x, 4),
            "cursor_y": round(cursor_y, 4),
            "action": action,
            "landmarks": landmark_list if len(landmark_list) == 42 else [],
            "frame": frame_base64,
            "model_ready": model_ready,
        }

        await broadcast(json.dumps(payload))
        await asyncio.sleep(0.025)

    cap.release()


# ====================== WEBSOCKET ======================
async def handler(websocket):
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "save_data":
                await save_data(data["csv_base64"])
            elif msg_type == "train":
                asyncio.create_task(train_model(
                    csv_base64=data.get("csv_base64"),
                    epochs=data.get("epochs", 30),
                    lr=data.get("lr", 0.001),
                    test_split=data.get("test_split", 0.2),
                ))
            elif msg_type == "get_dataset":
                await send_dataset_update()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(websocket)


async def main():
    print("Gesture Lab Server Running → ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await camera_loop()


if __name__ == "__main__":
    asyncio.run(main())
