import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  
from data.synthetic_dataset import SyntheticDotDataset
from models.tinyCNN import TinyCNNRegressor  

# Hyperparameters
epochs = 10
batch_size = 64
lr = 1e-3

# Data
train_dataset = SyntheticDotDataset(num_sequences=500, frames_per_seq=10)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model + Loss + Optimizer
model = TinyCNNRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
