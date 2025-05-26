import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DotDataset 
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.tinyCNN import TinyCNNRegressor

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
        transform=None
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train'))

train_loader = get_loader('train')
val_loader = get_loader('val')
test_loader = get_loader('test')

# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
model = TinyCNNRegressor().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

# ---------------------------
# Evaluation (on test set)
# ---------------------------
model.eval()
mse_list = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        mse = criterion(outputs, labels)
        mse_list.append(mse.item())

print(f"Test MSE: {sum(mse_list) / len(mse_list):.4f}")

# Save the trained model
torch.save(model.state_dict(), "tinycnn_trained.pth")


