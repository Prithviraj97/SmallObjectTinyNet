import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic_dataset import SyntheticDotDataset
from models.tinyCNN import TinyCNNRegressor

# Load model
model = TinyCNNRegressor()
model.load_state_dict(torch.load("tinyCNN_model.pth"))
model.eval()

# Load test data
test_dataset = SyntheticDotDataset(num_sequences=5, frames_per_seq=20)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Lambda for fusion
lambda_cnn = 0.6  # 0.0 = only physics, 1.0 = only CNN

cnn_predictions = []
fused_predictions = []
true_positions = []

# We manually track previous positions for physics
previous_position = None
pre_previous_position = None

with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        # CNN prediction
        cnn_output = model(img).squeeze().cpu()
        true_pos = label.squeeze().cpu()

        # Save CNN and true predictions
        cnn_predictions.append(np.array(cnn_output.tolist()))
        true_positions.append(np.array(true_pos.tolist()))

        # Physics-based prediction (if possible)
        if previous_position is not None and pre_previous_position is not None:
            velocity = previous_position - pre_previous_position
            physics_pred = previous_position + velocity
        else:
            physics_pred = cnn_output  # fallback to CNN for first two frames

        # Fusion
        fused_pred = lambda_cnn * cnn_output + (1 - lambda_cnn) * physics_pred
        fused_predictions.append(np.array(fused_pred.tolist()))

        # Update state
        pre_previous_position = previous_position
        previous_position = true_pos

# Convert lists to arrays
cnn_predictions = np.stack(cnn_predictions)
fused_predictions = np.stack(fused_predictions)
true_positions = np.stack(true_positions)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], label="True", c="blue")
plt.plot(cnn_predictions[:, 0], cnn_predictions[:, 1], label="CNN", c="red", linestyle='--')
plt.plot(fused_predictions[:, 0], fused_predictions[:, 1], label="Fused", c="green", linestyle=':')
plt.title("Trajectory: True vs CNN vs Fused")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
plt.savefig("trajectory_plot.png")
