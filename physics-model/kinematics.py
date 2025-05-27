import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from torchvision import transforms
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.dataset import DotDataset
# Load test dataset
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
test_dataset = DotDataset(
    image_dir="synthetic_dataset/test/images",
    label_file="synthetic_dataset/test/labels.csv",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate Physics Tracker
ground_truth_positions = []
predicted_positions = []

# We need at least two past frames to start
buffer = []

for i, (image, target) in enumerate(test_loader):
    target = target.squeeze()
    ground_truth_positions.append(target)

    # if len(buffer) < 2:
    #     # Not enough history to predict
    #     prediction = target
    # else:
    #     # Physics: x_t = 2x_{t-1} - x_{t-2}
    #     prediction = 2 * np.array(buffer[-1]) - np.array(buffer[-2])

    if len(buffer) < 2:
        prediction = target
    else:
        prev1 = buffer[-1].tolist() if hasattr(buffer[-1], "tolist") else buffer[-1]
        prev2 = buffer[-2].tolist() if hasattr(buffer[-2], "tolist") else buffer[-2]
        # Physics: x_t = 2x_{t-1} - x_{t-2}
        prediction = [2 * p1 - p2 for p1, p2 in zip(prev1, prev2)]
        prediction = torch.tensor(prediction, dtype=target.dtype)

    predicted_positions.append(prediction.tolist())
    buffer.append(target)

# Convert to arrays
# ground_truth_positions = np.array(ground_truth_positions)
# predicted_positions = np.array(predicted_positions)
ground_truth_positions = np.array([t.tolist() if hasattr(t, "tolist") else t for t in ground_truth_positions])
predicted_positions = np.array(predicted_positions)

# Compute MSE
mse = np.mean((ground_truth_positions - predicted_positions)**2)
print(f"Physics Tracker MSE: {mse:.5f}")

# Plotting predicted vs true
# plt.figure(figsize=(6,6))
# plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 'g-', label='Ground Truth')
# plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], 'r--', label='Physics Prediction')
# plt.legend()
# plt.title("Physics Tracker vs Ground Truth")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.grid(True)
# plt.axis("equal")
# plt.tight_layout()
# plt.show()
plt.figure(figsize=(6, 6))

# Scatter plot for all points
plt.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], color='green', label='Ground Truth', s=30)
plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], color='red', label='Physics Prediction', s=30, marker='x')

# Optional: Draw lines through the points to show the trajectory
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], color='green', linestyle='-', alpha=0.5)
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], color='red', linestyle='--', alpha=0.5)

plt.legend()
plt.title("Physics Tracker vs Ground Truth")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()