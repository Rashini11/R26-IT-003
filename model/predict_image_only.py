import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import json
import os

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "model/image_only_model.pth"
LABEL_MAP_PATH = "model/label_map.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load label map
# -----------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

reverse_label_map = {value: key for key, value in label_map.items()}

print("Label Map:", label_map)

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Model
# -----------------------------
class ImageOnlyMobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, image):
        return self.cnn(image)

# -----------------------------
# Load model
# -----------------------------
model = ImageOnlyMobileNet(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100

    sea_state = reverse_label_map[predicted_class]

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n🌊 Prediction Result")
    print("-------------------")
    print("Image:", image_path)
    print("Predicted Sea State:", sea_state)
    print(f"Confidence: {confidence:.2f}%")

    print("\n📊 Class Probabilities:")
    print("-----------------------")

    for i, prob in enumerate(probabilities[0]):
        label = reverse_label_map[i]
        print(f"{label:12}: {prob.item()*100:.2f}%")

# -----------------------------
# Run (interactive)
# -----------------------------
while True:
    image_path = input("\nEnter image path (or type 'exit'): ")

    if image_path.lower() == "exit":
        print("Exiting...")
        break

    predict_image(image_path)