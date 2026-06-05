# 🖐️ Hand Gesture Mouse Controller

Control your mouse using hand gestures detected through your webcam — no hardware, no wires.  
Built with **MediaPipe** (landmark detection) + **PyTorch** (neural network classifier).

---

## 📌 What This Does

Your webcam watches your hand. MediaPipe extracts 21 landmarks (x, y coordinates = 42 numbers). A trained neural network classifies which gesture you're doing. The app then moves your mouse or fires a click — all in real time.

```
Webcam → MediaPipe (21 landmarks × x,y) → GestureNet (PyTorch) → Mouse Action
```

---

## 🖐️ Supported Gestures

| Key | Gesture      | Action                 |
| --- | ------------ | ---------------------- |
| `0` | move         | Move mouse cursor      |
| `1` | left_click   | Left click             |
| `2` | right_click  | Right click            |
| `3` | double_click | Double click           |
| `4` | screenshot   | Take & save screenshot |

---

## 📁 Project Structure

```
gesture-mouse/
│
├── collect_data.py       # Step 1 — Record gesture samples via webcam
├── gesture_data.csv      # Collected dataset (1253 samples, 5 classes)
├── train.py              # Step 2 — Train PyTorch neural network
├── gesture_model.pth     # Trained model weights
├── label_encoder.pkl     # Sklearn LabelEncoder (maps class index → gesture label)
└── main.py               # Step 3 — Run live gesture mouse control
```

> `test_mouse3.py` is a standalone Windows script to verify cursor movement via `ctypes`.

---

## 🧠 How Each Part Works

### 1. `collect_data.py` — Data Collection

- Opens webcam and runs **MediaPipe Hands**
- Detects 21 hand landmarks (wrist + 4 joints per finger)
- Each frame = 42 floats (x, y for each landmark), normalized 0–1 by MediaPipe
- Press a gesture key (`0`–`4`), then `SPACE` to start recording
- Auto-stops at 250 samples per gesture
- Appends rows to `gesture_data.csv`

**What to do:** Run this once per gesture. Collected ~250 samples each = **1253 total rows**.

### 2. `train.py` — Model Training

- Reads `gesture_data.csv`
- Input: 42 landmark features | Output: 5 gesture classes
- Architecture: `42 → 128 → 64 → 5` (ReLU + Dropout)
- Trains for 50 epochs with Adam optimizer, CrossEntropyLoss
- Saves `gesture_model.pth` and `label_encoder.pkl`

**What to do:** Run after collecting data. Takes ~30 seconds on CPU.

### 3. `main.py` — Live Control

- Loads the trained model and label encoder
- Each webcam frame → MediaPipe → 42 landmarks → `GestureNet.predict()`
- If **confidence > 85%**, the gesture action fires
- Mouse movement uses `pynput` (smooth, no OS restrictions on Windows)
- Clicks use `pynput`, double-click uses `pyautogui`, screenshot saves as PNG

**What to do:** Run this to use the system. Press `Q` to quit.

---

## ⚙️ The Neural Network

```
Input (42)  →  Linear(128)  →  ReLU  →  Dropout(0.3)
            →  Linear(64)   →  ReLU  →  Dropout(0.2)
            →  Linear(5)    →  Softmax  →  Predicted Gesture
```

- Confidence threshold: **0.85** (actions only fire when model is sure)
- Action cooldown: **0.5s** (prevents repeated accidental clicks)

---

## 🚀 Setup & Usage

### Install dependencies

```bash
pip install mediapipe opencv-python torch pyautogui pynput scikit-learn pandas numpy
```

### Step-by-step

```bash
# 1. Collect gesture data (run for each gesture 0–4)
python collect_data.py

# 2. Train the model
python train.py

# 3. Run the mouse controller
python main.py
```

### Web UI: Data Collection + Playground

The `demo_ui.html` page now contains:

- Data Collection tab: records 42 landmark features from live WebSocket frames and exports CSV.
- Playground tab: visual gesture sandbox (cursor, zoom, scroll, highlight effects).

Run these in separate terminals:

```bash
# Terminal 1: start the websocket/camera backend
python demo_server.py

# Terminal 2: serve the UI
python -m http.server 8000
```

Then open `http://localhost:8000/demo_ui.html`.

---

## 📊 Dataset

| Gesture          | Samples  |
| ---------------- | -------- |
| move (0)         | 251      |
| left_click (1)   | 251      |
| right_click (2)  | 250      |
| double_click (3) | 250      |
| screenshot (4)   | 251      |
| **Total**        | **1253** |

---

## 🖥️ Platform Notes

- Developed and tested on **Windows**
- `test_mouse3.py` uses `ctypes.windll` — Windows only
- `main.py` uses `pynput` for mouse control which works cross-platform; replace `pyautogui.size()` with platform alternatives on Linux/Mac if needed

---

## 🔧 Troubleshooting

| Problem            | Fix                                              |
| ------------------ | ------------------------------------------------ |
| Mouse doesn't move | Check `pyautogui.FAILSAFE = False` is set        |
| Low accuracy       | Collect more samples, ensure consistent lighting |
| Hand not detected  | Improve lighting, keep hand fully in frame       |
| Clicks too fast    | Increase `COOLDOWN` in `main.py`                 |

---

## 📌 Future Ideas

- Add scroll gesture
- Support two-hand gestures
- Retrain with more diverse data (lighting, hand sizes)
- Package as `.exe` with PyInstaller
