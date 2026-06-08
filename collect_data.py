import cv2
import csv
import os
from matplotlib.pylab import rint
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils

GESTURES = {
    '0': 'move',
    '1': 'left_click',
    '2': 'right_click',
    '3': 'double_click',
    '4': 'screenshot',
    '5': 'scroll_up',      
    '6': 'scroll_down',
}

OUTPUT_FILE = 'gesture_data.csv'
TARGET = 250  # samples per gesture

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y']] + ['label']
        writer.writerow(header)

def main():
    cap = cv2.VideoCapture(0)
    current_gesture = None
    recording = False
    count = 0

    print("=== Data Collection ===")
    print("Press 0-6 to select gesture, SPACE to start/stop, Q to quit")
    for k, v in GESTURES.items():
        print(f"  {k} = {v}")

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

        # Save to CSV if recording and hand detected
        if recording and len(landmark_list) == 42:
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmark_list + [current_gesture])
            count += 1

            # Auto stop at target
            if count >= TARGET:
                recording = False
                print(f"✓ {TARGET} samples collected for {GESTURES[current_gesture]}")

        # --- UI overlay ---
        h, w = frame.shape[:2]

        # Top bar background
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

        # Gesture name
        gesture_name = GESTURES.get(current_gesture, 'None selected')
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Recording status
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        status_text = "● RECORDING" if recording else "■ STOPPED"
        cv2.putText(frame, status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Progress bar
        if current_gesture is not None:
            progress = int((count / TARGET) * (w - 20))
            cv2.rectangle(frame, (10, 65), (w - 10, 75), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 65), (10 + progress, 75), (0, 255, 0), -1)

        # Sample counter (big, top right)
        cv2.putText(frame, f"{count}/{TARGET}", (w - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in GESTURES:
            current_gesture = chr(key)
            recording = False
            count = 0
            print(f"Selected: {GESTURES[current_gesture]} — press SPACE to record")
        elif key == ord(' '):
            if current_gesture is None:
                print("Press 0-6 to select gesture, SPACE to start/stop, Q to quit")
            else:
                recording = not recording
                if recording:
                    print(f"Recording {GESTURES[current_gesture]}...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()