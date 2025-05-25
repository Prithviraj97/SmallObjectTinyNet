import torch
import matplotlib.pyplot as plt
import numpy as np

# Load test dataset
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate Physics Tracker
ground_truth_positions = []
predicted_positions = []

# We need at least two past frames to start
buffer = []

for i, (image, target) in enumerate(test_loader):
    target = target.squeeze().numpy()
    ground_truth_positions.append(target)

    if len(buffer) < 2:
        # Not enough history to predict
        prediction = target
    else:
        # Physics: x_t = 2x_{t-1} - x_{t-2}
        prediction = 2 * np.array(buffer[-1]) - np.array(buffer[-2])

    predicted_positions.append(prediction.tolist())
    buffer.append(target)

# Convert to arrays
ground_truth_positions = np.array(ground_truth_positions)
predicted_positions = np.array(predicted_positions)

# Compute MSE
mse = np.mean((ground_truth_positions - predicted_positions)**2)
print(f"Physics Tracker MSE: {mse:.5f}")

# Plotting predicted vs true
plt.figure(figsize=(6,6))
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'g-', label='Ground Truth')
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r--', label='Physics Prediction')
plt.legend()
plt.title("Physics Tracker vs Ground Truth")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
