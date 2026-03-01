import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── 1. Config ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 2. Load & Preprocess ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                        # converts to [0,1] float tensor
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean & std
])

full_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 90/10 train/validation split
val_size   = int(0.1 * len(full_train))
train_set, val_set = random_split(full_train, [len(full_train) - val_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

# ── 3. Build CNN Model ────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3),   # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 26x26 -> 13x13

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 11x11 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)                 # 10 classes
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN().to(DEVICE)
print(model)

# ── 4. Compile ────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── 5. Train ──────────────────────────────────────────────────────────────────
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        correct      += (out.argmax(1) == y).sum().item()
        total        += y.size(0)
    train_losses.append(running_loss / total)
    train_accs.append(correct / total)

    # --- Validation ---
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = criterion(out, y)
            running_loss += loss.item() * X.size(0)
            correct      += (out.argmax(1) == y).sum().item()
            total        += y.size(0)
    val_losses.append(running_loss / total)
    val_accs.append(correct / total)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

# ── 6. Evaluate on Test Set ───────────────────────────────────────────────────
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out  = model(X)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_acc = correct / total
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ── 7. Plot Training Curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_accs, label="Train")
axes[0].plot(val_accs,   label="Validation")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(train_losses, label="Train")
axes[1].plot(val_losses,   label="Validation")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")

# ── 8. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — MNIST Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ── 9. Visualize Sample Predictions ──────────────────────────────────────────
all_labels_arr = np.array(all_labels)
all_preds_arr  = np.array(all_preds)

raw_test = datasets.MNIST(root="./data", train=False, download=False,
                          transform=transforms.ToTensor())

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for digit in range(10):
    idxs = np.where(all_labels_arr == digit)[0][:3]
    for row, idx in enumerate(idxs):
        img, _ = raw_test[idx]
        ax = axes[row, digit]
        ax.imshow(img.squeeze(), cmap="gray")
        pred  = all_preds_arr[idx]
        color = "green" if pred == digit else "red"
        ax.set_title(f"P:{pred}", color=color, fontsize=8)
        ax.axis("off")

plt.suptitle("Sample Predictions (green=correct, red=wrong)", y=1.02)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Saved: sample_predictions.png")

# ── 10. Save Model ────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved: mnist_cnn.pth")
