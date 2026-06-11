import cv2
import csv
import os
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

# LEFT HAND TEST DATASET
OUTPUT_FILE = 'left_hand_test.csv'
TARGET = 50

# Create CSV file with header if it doesn't exist
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

    print("=" * 60)
    print("LEFT HAND TEST DATA COLLECTION")
    print("=" * 60)
    print("IMPORTANT: USE ONLY YOUR LEFT HAND")
    print(f"Output File : {OUTPUT_FILE}")
    print(f"Target      : {TARGET} samples per gesture")
    print()
    print("Press 0-6 to select gesture")
    print("SPACE = Start/Stop recording")
    print("Q = Quit")
    print()

    for k, v in GESTURES.items():
        print(f"{k} = {v}")

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

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

        # Save sample
        if recording and len(landmark_list) == 42:

            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmark_list + [current_gesture])

            count += 1

            if count >= TARGET:
                recording = False
                print(
                    f"✓ {TARGET} samples collected for "
                    f"{GESTURES[current_gesture]}"
                )

        # ---------------- UI ----------------

        h, w = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, 0),
            (w, 80),
            (0, 0, 0),
            -1
        )

        gesture_name = GESTURES.get(
            current_gesture,
            'None selected'
        )

        cv2.putText(
            frame,
            f"LEFT HAND TEST",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        status_color = (
            (0, 255, 0)
            if recording
            else (0, 0, 255)
        )

        status_text = (
            "● RECORDING"
            if recording
            else "■ STOPPED"
        )

        cv2.putText(
            frame,
            status_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        if current_gesture is not None:

            progress = int(
                (count / TARGET) * (w - 20)
            )

            cv2.rectangle(
                frame,
                (10, h - 20),
                (w - 10, h - 10),
                (50, 50, 50),
                -1
            )

            cv2.rectangle(
                frame,
                (10, h - 20),
                (10 + progress, h - 10),
                (0, 255, 0),
                -1
            )

        cv2.putText(
            frame,
            f"{count}/{TARGET}",
            (w - 140, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "LEFT HAND TEST DATA COLLECTION",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key != 255:
            try:
                pressed = chr(key)

                if pressed in GESTURES:

                    current_gesture = pressed
                    recording = False
                    count = 0

                    print(
                        f"\nSelected: "
                        f"{GESTURES[current_gesture]}"
                    )
                    print(
                        "Press SPACE to start recording"
                    )

            except:
                pass

        elif key == ord(' '):
            pass

        if key == ord(' '):

            if current_gesture is None:

                print(
                    "Select gesture (0-6) first."
                )

            else:

                recording = not recording

                if recording:
                    print(
                        f"Recording "
                        f"{GESTURES[current_gesture]}..."
                    )

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinished.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()