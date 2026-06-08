import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())

    return np.array(y_true), np.array(y_pred)


def print_confusion_matrix(matrix, labels):
    width = max(14, max(len(label) for label in labels) + 2)

    print("\nCONFUSION MATRIX")
    print("Rows = actual labels, Columns = predicted labels\n")
    print(f"{'Actual \\ Pred':<{width}}", end="")
    for label in labels:
        print(f"{label:>{width}}", end="")
    print()

    for label, row in zip(labels, matrix):
        print(f"{label:<{width}}", end="")
        for value in row:
            print(f"{value:>{width}}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate the hand gesture classifier using train/test accuracy, confusion matrix, precision, recall, and F1-score."
    )
    parser.add_argument("--data", default=DATA_FILE, help="Path to gesture CSV file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y_labels = load_dataset(args.data)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_labels = [str(label) for label in label_encoder.classes_]
    display_labels = [
        f"{label} ({GESTURE_NAMES.get(label, 'unknown')})" for label in class_labels
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_loader, test_loader = make_loaders(
        X_train,
        X_test,
        y_train,
        y_test,
        args.batch_size,
    )

    model = GestureNet(input_size=X.shape[1], output_size=len(class_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("VALIDATION SETUP")
    print(f"Dataset file       : {args.data}")
    print(f"Total samples      : {len(X)}")
    print(f"Training samples   : {len(X_train)}")
    print(f"Testing samples    : {len(X_test)}")
    print(f"Classes            : {', '.join(display_labels)}")
    print(f"Epochs             : {args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

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
            avg_loss = total_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1:>3}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}%"
            )

    train_true, train_pred = evaluate(model, train_loader)
    test_true, test_pred = evaluate(model, test_loader)

    train_acc = accuracy_score(train_true, train_pred) * 100
    test_acc = accuracy_score(test_true, test_pred) * 100

    print("\nFINAL ACCURACY")
    print(f"Train accuracy : {train_acc:.2f}%")
    print(f"Test accuracy  : {test_acc:.2f}%")

    matrix = confusion_matrix(test_true, test_pred, labels=range(len(class_labels)))
    print_confusion_matrix(matrix, display_labels)

    print("\nPRECISION, RECALL AND F1-SCORE")
    print(
        classification_report(
            test_true,
            test_pred,
            labels=list(range(len(class_labels))),
            target_names=display_labels,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
