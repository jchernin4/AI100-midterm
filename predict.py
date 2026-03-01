import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import sys

# ── Model definition (must match mnist_cnn.py) ────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
model.eval()

# ── Load & preprocess image ───────────────────────────────────────────────────
def preprocess(image_path):
    img = Image.open(image_path).convert("L")  # convert to grayscale

    # MNIST is white digit on black background — invert if your image is opposite
    # Comment this out if your digit is already white-on-black
    img = ImageOps.invert(img)

    img = img.resize((28, 28))                  # resize to 28x28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(img).unsqueeze(0)          # add batch dimension

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image_path):
    tensor = preprocess(image_path).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()

    # Show image + result
    img = Image.open(image_path).convert("L").resize((28, 28))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Your image (resized to 28x28)")
    axes[0].axis("off")

    axes[1].bar(range(10), probs.cpu().numpy())
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title(f"Prediction: {pred}  ({probs[pred]*100:.1f}% confident)")

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()
    print(f"Predicted digit: {pred}  (confidence: {probs[pred]*100:.1f}%)")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py my_digit.png")
        sys.exit(1)
    predict(sys.argv[1])
