import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


DATA_FILE = "gesture_data.csv"

GESTURE_NAMES = {
    "0": "move",
    "1": "left_click",
    "2": "right_click",
    "3": "double_click",
    "4": "screenshot",
    "5": "scroll_up",
    "6": "scroll_down",
}


class GestureNet(nn.Module):
    def __init__(self, input_size=42, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(path):
    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].astype(str).values

    if X.shape[1] != 42:
        raise ValueError(f"Expected 42 feature columns, found {X.shape[1]}.")

    return X, y


def make_loaders(X_train, X_test, y_train, y_test, batch_size):
    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())

    return np.array(y_true), np.array(y_pred)


def print_confusion_matrix(matrix, labels):
    max_label_len = max(len(label) for label in labels)
    max_num_len = max(len(str(max(max(row) for row in matrix))), 3)
    width = max(max_label_len + 2, max_num_len + 2)

    print("\nCONFUSION MATRIX")
    print("Rows = actual labels, Columns = predicted labels\n")

    header = "Actual \\ Pred".rjust(width)
    print(header, end="")

    for label in labels:
        print(f"{label:>{width}}", end="")
    print()

    total_width = width * (len(labels) + 1)
    print("-" * total_width)

    for label, row in zip(labels, matrix):
        print(f"{label:>{width}}", end="")
        for value in row:
            print(f"{value:>{width}}", end="")
        print()

    print("-" * total_width)


# ---------------- SAVE FUNCTIONS ----------------

def save_confusion_matrix_png(matrix, labels):
    os.makedirs("validation", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(j, i, str(matrix[i][j]),
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig("validation/confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved: validation/confusion_matrix.png")


def save_accuracy_plot(epochs, train_acc, test_acc):
    os.makedirs("validation", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Testing Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("validation/accuracy.png", dpi=300)
    plt.close()

    print("Saved: validation/accuracy.png")


def save_classification_report_txt(report):
    os.makedirs("validation", exist_ok=True)

    with open("validation/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Saved: validation/classification_report.txt")


def format_confusion_matrix_text(matrix, labels):
    max_label_len = max(len(label) for label in labels)
    max_num_len = max(len(str(int(np.max(matrix)))), 3)
    width = max(max_label_len + 2, max_num_len + 2)

    lines = []
    lines.append("CONFUSION MATRIX")
    lines.append("Rows = actual labels, Columns = predicted labels")
    lines.append("")

    header = "Actual \\ Pred".rjust(width)
    header += "".join(f"{label:>{width}}" for label in labels)
    lines.append(header)
    lines.append("-" * (width * (len(labels) + 1)))

    for label, row in zip(labels, matrix):
        row_text = f"{label:>{width}}"
        row_text += "".join(f"{value:>{width}}" for value in row)
        lines.append(row_text)

    lines.append("-" * (width * (len(labels) + 1)))
    return "\n".join(lines)


def save_validation_results_txt(dataset_file, total_samples, train_samples, test_samples,
                               epochs, train_acc, test_acc, matrix, labels, report):
    os.makedirs("validation", exist_ok=True)

    content = [
        "VALIDATION SETUP",
        f"Dataset file     : {dataset_file}",
        f"Total samples    : {total_samples}",
        f"Training samples : {train_samples}",
        f"Testing samples  : {test_samples}",
        f"Epochs           : {epochs}",
        "",
        "FINAL RESULTS",
        f"Train Accuracy: {train_acc:.2f}%",
        f"Test Accuracy : {test_acc:.2f}%",
        "",
        format_confusion_matrix_text(matrix, labels),
        "",
        "PRECISION / RECALL / F1",
        report,
    ]

    with open("validation/validation_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    print("Saved: validation/validation_results.txt")


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y_labels = load_dataset(args.data)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    class_labels = [str(label) for label in label_encoder.classes_]
    display_labels = [
        f"{label} ({GESTURE_NAMES.get(label, 'unknown')})"
        for label in class_labels
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_loader, test_loader = make_loaders(
        X_train, X_test, y_train, y_test, args.batch_size
    )

    model = GestureNet(input_size=X.shape[1], output_size=len(class_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_acc_history = []
    test_acc_history = []
    epochs_history = []

    print("VALIDATION SETUP")
    print(f"Dataset file     : {args.data}")
    print(f"Total samples    : {len(X)}")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Epochs           : {args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            train_true, train_pred = evaluate(model, train_loader)
            test_true, test_pred = evaluate(model, test_loader)

            train_acc = accuracy_score(train_true, train_pred) * 100
            test_acc = accuracy_score(test_true, test_pred) * 100

            epochs_history.append(epoch + 1)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {total_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}%"
            )

    train_true, train_pred = evaluate(model, train_loader)
    test_true, test_pred = evaluate(model, test_loader)

    train_acc = accuracy_score(train_true, train_pred) * 100
    test_acc = accuracy_score(test_true, test_pred) * 100

    print("\nFINAL RESULTS")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy : {test_acc:.2f}%")

    matrix = confusion_matrix(
        test_true,
        test_pred,
        labels=range(len(class_labels))
    )

    print_confusion_matrix(matrix, display_labels)

    save_confusion_matrix_png(matrix, display_labels)
    save_accuracy_plot(epochs_history, train_acc_history, test_acc_history)

    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(len(class_labels))),
        target_names=display_labels,
        digits=4,
        zero_division=0,
    )

    print("\nPRECISION / RECALL / F1")
    print(report)

    save_classification_report_txt(report)


if __name__ == "__main__":
    main()