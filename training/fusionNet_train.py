import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fusionNet import FusionNet, fusion_loss
from training.dataset import DotDataset

# ---------------------------
# Configuration
# ---------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # converts [0,255] to [0.0,1.0]
])

# ---------------------------
# Dataset & DataLoader
# ---------------------------
def get_loader(split):
    dataset = DotDataset(
        image_dir=f'synthetic_dataset/{split}/images',
        label_file=f'synthetic_dataset/{split}/labels.csv',
        transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train'))

train_loader = get_loader('train')
val_loader = get_loader('val')
test_loader = get_loader('test')

# Initialize model
fusion_model = FusionNet()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

# Training loop
epochs = 10
lambda_values = []
loss_values = []

for epoch in range(epochs):
    total_loss = 0.0
    for img, pos1, pos2, target in train_loader:
        optimizer.zero_grad()
        pred, _, _, lambda_t = fusion_model(img, pos1, pos2)
        loss = fusion_loss(pred, target, lambda_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    lambda_values.append(fusion_model.lambda_t.item())
    loss_values.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Lambda={fusion_model.lambda_t.item():.4f}")