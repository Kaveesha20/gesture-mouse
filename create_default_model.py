import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os

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

def create_default_model():
    print("\n" + "="*60)
    print("CREATING DEFAULT MODEL")
    print("="*60)
    
    # Check if data file exists
    if not os.path.exists('gesture_data.csv'):
        print("\n[ERROR] gesture_data.csv not found!")
        print("\nPlease collect data first using the data collection tool:")
        print("  1. Run: python collect_data.py")
        print("  2. Record samples for each gesture (0-6)")
        print("  3. Then run this script again")
        return False
    
    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv('gesture_data.csv')
    print(f"      Loaded {len(df)} samples")
    
    if len(df) < 10:
        print("\n[ERROR] Need at least 10 samples to train!")
        print(f"      Current samples: {len(df)}")
        print("\nPlease collect more data using collect_data.py")
        return False
    
    # Check label distribution
    print("\n[2/5] Checking label distribution...")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        gesture_names = {
            '0': 'Move', '1': 'Left Click', '2': 'Right Click',
            '3': 'Double Click', '4': 'Screenshot', '5': 'Scroll Up', '6': 'Scroll Down'
        }
        print(f"      Gesture {label} ({gesture_names.get(label, 'Unknown')}): {count} samples")
    
    # Prepare data
    print("\n[3/5] Preparing data for training...")
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"      Feature shape: {X.shape}")
    print(f"      Number of classes: {len(le.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples: {len(X_test)}")
    
    # Create data loaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # Create and train model
    print("\n[4/5] Training model...")
    model = GestureNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                output = model(xb)
                _, predicted = torch.max(output, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        val_acc = (correct / total) * 100
        avg_loss = total_loss / len(train_loader)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.1f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model and encoder
    print("\n[5/5] Saving model...")
    torch.save(model.state_dict(), 'gesture_model_default.pth')
    with open('label_encoder_default.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("\n" + "="*60)
    print("✓ DEFAULT MODEL CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModel saved as: gesture_model_default.pth")
    print(f"Encoder saved as: label_encoder_default.pkl")
    print(f"Final validation accuracy: {best_val_acc:.1f}%")
    
    # Test the model with a sample
    print("\n[TEST] Running a quick test prediction...")
    model.eval()
    sample = X_test[0:1]
    sample_tensor = torch.tensor(sample, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(sample_tensor)
        pred_class = torch.argmax(prediction, dim=1).item()
        confidence = torch.softmax(prediction, dim=1).max().item()
    
    actual_label = le.inverse_transform([y_test[0]])[0]
    predicted_label = le.inverse_transform([pred_class])[0]
    
    print(f"      Sample test - Actual: {actual_label}, Predicted: {predicted_label}, Confidence: {confidence*100:.1f}%")
    
    return True

def test_model():
    """Optional function to test the saved model"""
    print("\n" + "="*60)
    print("TESTING DEFAULT MODEL")
    print("="*60)
    
    if not os.path.exists('gesture_model_default.pth'):
        print("[ERROR] No default model found. Please create it first.")
        return False
    
    # Load model
    model = GestureNet()
    model.load_state_dict(torch.load('gesture_model_default.pth', weights_only=True))
    model.eval()
    
    # Load encoder
    with open('label_encoder_default.pkl', 'rb') as f:
        le = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Encoder loaded successfully")
    print(f"✓ Model supports gestures: {le.classes_}")
    
    return True

if __name__ == "__main__":
    # Create the default model
    success = create_default_model()
    
    if success:
        # Test the model
        test_model()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Run the server: python demo_server.py")
        print("2. Open index.html in your browser")
        print("3. Use Default Mode or switch to Customize Mode")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("FAILED TO CREATE DEFAULT MODEL")
        print("="*60)
        print("\nPlease ensure you have:")
        print("  1. Collected sufficient training data using collect_data.py")
        print("  2. Have at least 10 samples per gesture")
        print("  3. Run this script again")
        print("="*60 + "\n")