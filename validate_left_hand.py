import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------
# Model Definition
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

# --------------------------
# Load trained model
# --------------------------

print("Loading model...")

model = GestureNet()
model.load_state_dict(
    torch.load("gesture_model.pth", map_location=torch.device("cpu"))
)
model.eval()

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Model loaded.")

# --------------------------
# Load LEFT HAND test dataset
# --------------------------

DATA_FILE = "left_hand_test.csv"

df = pd.read_csv(DATA_FILE)

X = df.drop("label", axis=1).values.astype(np.float32)
y_labels = df["label"].astype(str).values

# Convert labels using original encoder
y_true = le.transform(y_labels)

# --------------------------
# Predict
# --------------------------

X_tensor = torch.tensor(X)

with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).numpy()

# --------------------------
# Metrics
# --------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\n==============================")
print("LEFT HAND VALIDATION RESULTS")
print("==============================")

print(f"\nSamples Tested : {len(y_true)}")
print(f"Accuracy       : {accuracy * 100:.2f}%")

# --------------------------
# Classification Report
# --------------------------

print("\nPRECISION / RECALL / F1\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=[str(c) for c in le.classes_],
    digits=4,
    zero_division=0,
)

print(report)

# --------------------------
# Confusion Matrix
# --------------------------

cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

# --------------------------
# Per-class Accuracy
# --------------------------

print("\nPER-GESTURE ACCURACY")

for i, label in enumerate(le.classes_):
    total = cm[i].sum()

    if total == 0:
        continue

    correct = cm[i][i]
    acc = correct / total * 100

    print(f"Gesture {label}: {acc:.2f}%")

print("\nDone.")