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

| Key | Gesture      | Hand Pose                        | Action                 |
| --- | ------------ | -------------------------------- | ---------------------- |
| `0` | move         | Index finger up                  | Move mouse cursor      |
| `1` | left_click   | Thumb + Index pinch              | Left click             |
| `2` | right_click  | Index + Pinky up                 | Right click            |
| `3` | double_click | All fingers bent (fist)          | Double click           |
| `4` | screenshot   | All fingers extended (open hand) | Take & save screenshot |
| `5` | scroll_up    | pinky up                         | Scroll up              |
| `6` | scroll_down  | Thumb up                         | Scroll down            |

---

## 🚀 Quick Start (UI only)

Everything — **data collection, training, and live mouse control** — runs from the web UI. You do **not** need to run `collect_data.py`, `train.py`, or `main.py` separately.

### Install dependencies

```bash
pip install mediapipe opencv-python torch pyautogui pynput scikit-learn pandas numpy websockets
```

### Run one command

```bash
python run_ui.py
```

This starts the backend server, serves the UI, and opens your browser to `http://localhost:8000/demo_ui.html`.

### Use the 3 tabs in the browser

| Tab | Replaces | What to do |
| --- | -------- | ---------- |
| **Data Collection** | `collect_data.py` | Select gesture 0–6 → choose **Default Mode** or **Custom Mode** → **Start Recording** → collect ~250 samples → **Save to Server** |
| **Train Model** | `train.py` | Click **Train & Update Model** — trains on the active dataset (`gesture_data.csv` or `custom_gesture_data.csv`), retrains, and hot-reloads the model |
| **Playground** | `main.py` | Test gestures visually; the server also controls your real mouse while running |

**Tip:** Enable **Retrain model after saving new data** on the Data Collection tab to automatically train after each save.

**Custom mode:** Switching to **Custom Mode** resets `custom_gesture_data.csv` before new samples are collected, while `gesture_data.csv` stays intact.

---

## 📁 Project Structure

```
gesture-mouse/
│
├── run_ui.py             # ← Start here (one command)
├── demo_server.py        # WebSocket backend: camera, inference, training, mouse control
├── demo_ui.html          # Web UI: collect, train, playground
├── gesture_data.csv      # Dataset (grows as you save from the UI)
├── gesture_model.pth     # Trained model weights (created/updated by UI training)
├── label_encoder.pkl     # Label encoder (created/updated by UI training)
│
├── collect_data.py       # Legacy CLI collector (optional)
├── train.py              # Legacy CLI trainer (optional)
└── main.py               # Legacy CLI live control (optional)
```

---

## 🧠 How It Works

### Data Collection (UI)

- Server streams webcam frames + 42 landmark features over WebSocket
- UI records samples per gesture label into a buffer
- **Save to Server** appends rows to `gesture_data.csv`

### Training (UI)

- Trains `GestureNet` on the full server dataset
- Architecture: `42 → 128 → 64 → 7` (ReLU + Dropout)
- Saves `gesture_model.pth` and `label_encoder.pkl`
- Model is hot-reloaded for live inference immediately

### Live Control (server + Playground tab)

- Each frame → MediaPipe → 42 landmarks → `GestureNet.predict()`
- If **confidence > 85%**, the gesture action fires
- Mouse movement via `ctypes` (Windows), clicks via `pynput`
- Scroll via `pyautogui.scroll()`
- Action cooldown: **0.5s**

---

## ⚙️ The Neural Network

```
Input (42)  →  Linear(128)  →  ReLU  →  Dropout(0.3)
            →  Linear(64)   →  ReLU  →  Dropout(0.2)
            →  Linear(7)    →  Softmax  →  Predicted Gesture
```


---

## 🖥️ Platform Notes

- Developed and tested on **Windows**
- `demo_server.py` uses `ctypes.windll` for cursor movement — Windows only
- Legacy `main.py` is cross-platform for mouse clicks via `pynput`

---

## 🔧 Troubleshooting

| Problem              | Fix                                                        |
| -------------------- | ---------------------------------------------------------- |
| UI won't connect     | Make sure `python run_ui.py` is running                    |
| Model: not trained   | Collect data in UI, then use Train Model tab               |
| Mouse doesn't move   | Keep `demo_server.py` running (started by `run_ui.py`)     |
| Low accuracy         | Collect more samples per gesture, improve lighting         |
| Hand not detected    | Improve lighting, keep hand fully in frame                 |
| Clicks too fast      | Increase `COOLDOWN` in `demo_server.py`                    |
| Scroll not working   | Make sure model was retrained after adding gestures 5 & 6  |

---

## 📌 Future Ideas

- Support two-hand gestures
- Add drag gesture
- Package as `.exe` with PyInstaller
