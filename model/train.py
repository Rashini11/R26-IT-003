import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset.csv")

# 🔥 Reduce dataset size (IMPORTANT)
df = df.sample(n=2000, random_state=42)

print("Dataset size:", len(df))

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Encode labels
# -----------------------------
labels = df['label'].unique()
label_map = {label: i for i, label in enumerate(labels)}
df['label'] = df['label'].map(label_map)

print("Labels:", label_map)

# -----------------------------
# Dataset class
# -----------------------------
class SeaDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = row['image_path']
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        # Sensor data
        sensor = torch.tensor(
            [row['wave_height'], row['wind_speed']],
            dtype=torch.float32
        )

        # Label
        label = torch.tensor(row['label'], dtype=torch.long)

        return img, sensor, label

# -----------------------------
# DataLoader (FASTER)
# -----------------------------
dataset = SeaDataset(df)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,   # 🔥 increased batch size
    shuffle=True
)

# -----------------------------
# Model
# -----------------------------
class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(512, 128)

        self.sensor_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor_net(sensor)

        combined = torch.cat((img_feat, sensor_feat), dim=1)
        output = self.fc(combined)

        return output

model = MultiModalModel(num_classes=len(labels))

# -----------------------------
# Training setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training loop
# -----------------------------
EPOCHS = 3

for epoch in range(EPOCHS):
    total_loss = 0

    print(f"\n🚀 Starting Epoch {epoch+1}")

    for i, (img, sensor, label) in enumerate(loader):
        optimizer.zero_grad()

        output = model(img, sensor)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 5 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} completed, Total Loss: {total_loss:.4f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/sea_model.pth")

print("\n✅ Model trained and saved successfully!")