import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic_dataset import SyntheticDotDataset
from models.tinyCNN import TinyCNNRegressor

# Setup
model = TinyCNNRegressor()
model.load_state_dict(torch.load("tinyCNN_model.pth"))
model.eval()

test_dataset = SyntheticDotDataset(num_sequences=1, frames_per_seq=20)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Ground truth and CNN predictions
true_positions = []
cnn_predictions = []

with torch.no_grad():
    for img, label in test_loader:
        output = model(img).squeeze().cpu()
        cnn_predictions.append(output)
        true_positions.append(label.squeeze().cpu())

cnn_predictions = torch.stack(cnn_predictions)
true_positions = torch.stack(true_positions)

# Build physics predictions
physics_predictions = []
for i in range(len(true_positions)):
    if i < 2:
        physics_predictions.append(cnn_predictions[i])  # fallback
    else:
        velocity = true_positions[i-1] - true_positions[i-2]
        predicted = true_positions[i-1] + velocity
        physics_predictions.append(predicted)

physics_predictions = torch.stack(physics_predictions)

# Sweep lambda and calculate MSE at each lambda value
lambdas = np.linspace(0, 1, 21)
mse_values = []

for lam in lambdas:
    fused = lam * cnn_predictions + (1 - lam) * physics_predictions
    mse = mean_squared_error(np.array(true_positions.tolist()), np.array(fused.tolist()))
    mse_values.append(mse)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(lambdas, mse_values, marker='o')
plt.title("MSE vs. Lambda (Fusion Weight)")
plt.xlabel("Lambda (Weight for CNN Prediction)")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()
