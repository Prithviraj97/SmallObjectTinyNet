import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic_dataset import SyntheticTrackingDataset
from models.fusionNet import FusionNet, fusion_loss

# Define synthetic dataset
class SyntheticTrackingDataset(Dataset):
    def __init__(self, num_sequences=100, seq_len=10, image_size=64, speed=2):
        self.samples = []
        for _ in range(num_sequences):
            x = np.random.randint(5, image_size - 5)
            y = np.random.randint(5, image_size - 5)
            vx = np.random.choice([-speed, speed])
            vy = np.random.choice([-speed, speed])

            sequence = []
            for t in range(seq_len):
                img = np.zeros((image_size, image_size), dtype=np.float32)
                cx = np.clip(int(x + vx * t), 0, image_size - 1)
                cy = np.clip(int(y + vy * t), 0, image_size - 1)
                img[cy, cx] = 1.0
                sequence.append((img, np.array([cx, cy], dtype=np.float32)))
            for t in range(2, seq_len):
                self.samples.append((
                    sequence[t][0],     # x_t image
                    sequence[t-1][1],   # x(t-1) position
                    sequence[t-2][1],   # x(t-2) position
                    sequence[t][1]      # target
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, pos1, pos2, target = self.samples[idx]
        return torch.tensor(img).unsqueeze(0), torch.tensor(pos1), torch.tensor(pos2), torch.tensor(target)

# Create dataset and dataloader
dataset = SyntheticTrackingDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
fusion_model = FusionNet()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

# Training loop
epochs = 10
lambda_values = []
loss_values = []

for epoch in range(epochs):
    total_loss = 0.0
    for img, pos1, pos2, target in dataloader:
        optimizer.zero_grad()
        pred, _, _, lambda_t = fusion_model(img, pos1, pos2)
        loss = fusion_loss(pred, target, lambda_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    lambda_values.append(fusion_model.lambda_t.item())
    loss_values.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Lambda={fusion_model.lambda_t.item():.4f}")

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FusionNet Loss Curve")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambda_values, label="Lambda(t)", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Lambda Value")
plt.title("Learned Fusion Weight λ(t) Over Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

