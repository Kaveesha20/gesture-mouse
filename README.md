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

---

## 📊 Results and Discussion

### Model Performance

The trained `GestureNet` model achieved **99.84% accuracy** on the validation dataset (1,850 test samples) with the following per-gesture performance:

| Gesture          | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Move (0)         | 99.01%    | 100%   | 99.50%   | 300     |
| Left Click (1)   | 100%      | 98.80% | 99.40%   | 250     |
| Right Click (2)  | 100%      | 100%   | 100%     | 250     |
| Double Click (3) | 100%      | 100%   | 100%     | 300     |
| Screenshot (4)   | 100%      | 100%   | 100%     | 250     |
| Scroll Up (5)    | 100%      | 100%   | 100%     | 250     |
| Scroll Down (6)  | 100%      | 100%   | 100%     | 250     |

**Weighted Average:** 99.84% accuracy with a macro-average F1-score of 0.9984.

### Inference Speed & Real-Time Performance

- **End-to-end latency:** ~30–50ms per frame (30 FPS webcam stream)
- **MediaPipe landmark extraction:** ~10–15ms
- **Neural network inference:** ~5–10ms
- **Mouse control execution:** ~5–10ms
- **Confidence threshold:** 85% (reduces false positives)
- **Action cooldown:** 0.5s (prevents rapid repeated actions)

### Discussion

1. **High Accuracy Across All Gestures**  
   The model exhibits excellent performance, with most gestures achieving 100% precision and recall. The "move" gesture shows a slightly lower precision (99.01%) but maintains perfect recall, indicating that false positives are minimal while all true positives are captured.

2. **Robust Gesture Discrimination**  
   The distinction between similar gestures (e.g., left_click vs. right_click, scroll_up vs. scroll_down) is well-learned. The neural network successfully leverages MediaPipe's 21-landmark hand poses to differentiate subtle hand configurations.

3. **Real-Time Feasibility**  
   With an average inference time of 30–50ms per frame, the system runs smoothly at 20–30 FPS, providing a responsive user experience. The 0.5s cooldown prevents accidental double-actions while remaining user-friendly.

4. **Limiting Factors**  
   - **Lighting conditions:** Low or inconsistent lighting reduces MediaPipe landmark accuracy.
   - **Hand occlusion:** Partial hand visibility (e.g., hand near frame edge) reduces detection reliability.
   - **Hand scale variability:** Distance from the camera affects landmark precision.
   - **Single-hand limitation:** Currently supports only one hand; gestures with both hands are not recognized.

5. **Practical Usability**  
   The 85% confidence threshold effectively balances responsiveness and false-positive reduction. Users can achieve smooth mouse control with natural hand movements, and the dual-mode system (Default/Custom) allows for personalized gesture training.

---

## 🎯 Novelty (Optional)

While gesture recognition is a well-established field, this project demonstrates several noteworthy aspects:

1. **End-to-End WebSocket-Based UI**  
   The integration of a real-time web interface with a Python backend via WebSocket provides an accessible, zero-install-for-users approach to gesture recognition. Users can collect data, retrain the model, and test gestures directly from their browser without CLI knowledge.

2. **Dual-Mode Data Collection**  
   The Default and Custom modes allow users to:
   - Use pre-trained models for immediate use
   - Collect personalized gesture data to adapt the model to individual hand characteristics
   - Seamlessly switch between modes without data contamination

3. **Hot Model Reloading**  
   The server dynamically reloads trained models without interrupting the live stream, enabling rapid iteration and testing cycles during data collection and training phases.

4. **Lightweight Neural Architecture**  
   The `42 → 128 → 64 → 7` architecture is deliberately compact, enabling sub-10ms inference on CPU and real-time performance without GPU acceleration.

5. **Practical Multi-Action System**  
   Unlike many academic gesture projects that recognize only isolated hand poses, this system maps gestures to concrete OS-level actions (mouse movement, clicks, scrolling, screenshots) with intelligent cooldown and confidence thresholding.

---

## 🎓 Conclusion

This project successfully demonstrates a **practical, real-time hand gesture recognition system** that bridges the gap between academic research and user-friendly application design. By leveraging MediaPipe for robust hand landmark detection and PyTorch for efficient gesture classification, we achieve 99.84% accuracy while maintaining sub-50ms inference latency.

### Key Achievements

✅ **High Accuracy:** 99.84% on validation set across 7 distinct gestures  
✅ **Real-Time Performance:** 30–50ms end-to-end latency at 20–30 FPS  
✅ **User-Friendly Interface:** No CLI required; full browser-based workflow  
✅ **Adaptive Learning:** Support for custom gesture training and personalization  
✅ **Practical Functionality:** Direct mouse and system control without external hardware  

### Implications & Impact

1. **Accessibility:** Provides a hardware-free, gesture-based interface for users who may benefit from alternative input methods.
2. **Research Foundation:** Serves as a solid baseline for further work in gesture recognition (e.g., two-hand gestures, dynamic gestures, custom action mapping).
3. **Production Readiness:** With appropriate optimization (GPU inference, compiled models), the system can scale to production use in gaming, accessibility tools, or AR/VR applications.

### Recommendations for Future Work

- **Two-Hand Gestures:** Extend MediaPipe to track both hands simultaneously for richer gesture vocabulary.
- **Temporal Modeling:** Incorporate motion sequences (e.g., LSTM) to recognize dynamic gestures like swipes and rotations.
- **Hand Pose Normalization:** Apply data augmentation and normalization techniques to improve robustness across hand sizes and orientations.
- **GPU Acceleration:** Optimize model inference with TorchScript or ONNX for faster processing on dedicated hardware.
- **Deployment:** Package as an executable or browser extension for broader accessibility.
