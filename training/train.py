import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  
from data.synthetic_dataset import SyntheticDotDataset
from models.tinyCNN import TinyCNNRegressor  

# Hyperparameters
epochs = 20
batch_size = 16
lr = 1e-3

# Data
train_dataset = SyntheticDotDataset(num_sequences=500, frames_per_seq=10)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Evaluation
test_dataset = SyntheticDotDataset(num_sequences=50, frames_per_seq=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
model = TinyCNNRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss_histroy = []

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
    train_loss_histroy.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "tinyCNN_model.pth")

'''
Evlaution and Visualization
'''
import numpy as np
model.eval()
predicted_positions = []
true_positions = []
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        predicted_positions.append(outputs.squeeze().numpy())
        true_positions.append(labels.squeeze().numpy())

predicted_positions = np.array(predicted_positions)
true_positions = np.array(true_positions)

# Visualize the predicted and true positions
plt.figure(figsize=(6, 6))
plt.scatter(true_positions[:, 0], true_positions[:, 1], label='True', c='blue', alpha=0.6)
plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted', c='red', marker='x')
plt.title("Predicted vs Actual Object Positions")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
