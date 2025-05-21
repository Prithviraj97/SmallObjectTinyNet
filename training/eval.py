import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic_dataset import SyntheticDotDataset
from models.tinyCNN import TinyCNNRegressor

# Load test data - 50 sequences of 10 frames each for total of 500 frames.
test_dataset = SyntheticDotDataset(num_sequences=50, frames_per_seq=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
model = TinyCNNRegressor()

# Load the saved model weights
model.load_state_dict(torch.load("tinyCNN_model.pth"))
# model.eval()

# # Evaluation
# predicted_positions = []
# true_positions = []
# with torch.no_grad():
#     for imgs, labels in test_loader:
#         outputs = model(imgs)
#         predicted_positions.append(np.squeeze(outputs))
#         true_positions.append(np.squeeze(labels))

# predicted_positions = np.stack(predicted_positions)
# true_positions = np.stack(true_positions)

# # Scatter Plot
# plt.figure(figsize=(6, 6))
# plt.scatter(true_positions[:, 0], true_positions[:, 1], label='True', c='blue', alpha=0.6)
# plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted', c='red', marker='x')
# plt.title("Predicted vs Actual Object Positions")
# plt.xlabel("x position")
# plt.ylabel("y position")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# model.eval()
# predicted_positions = []
# true_positions = []

# with torch.no_grad():
#     for img, label in test_loader:
#         output = model(img)
#         predicted_positions.append(output.squeeze().cpu().numpy())
#         true_positions.append(label.squeeze().cpu().numpy())

# # Convert list of arrays to 2D NumPy arrays (shape: [N, 2])
# predicted_positions = np.array(predicted_positions)
# true_positions = np.array(true_positions)

# # 🎯 Scatter Plot
# plt.figure(figsize=(6, 6))
# plt.scatter(true_positions[:, 0], true_positions[:, 1], label='True', c='blue', alpha=0.6)
# plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted', c='red', marker='x')
# plt.title("Predicted vs Actual Object Positions")
# plt.xlabel("x position")
# plt.ylabel("y position")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.show()

predicted_positions = []
true_positions = []

with torch.no_grad():
    for img, label in test_loader:
        output = model(img)
        predicted_positions.append(output.squeeze())
        true_positions.append(label.squeeze())

# Stack into [N, 2] tensors
predicted_positions = torch.stack(predicted_positions).cpu()
true_positions = torch.stack(true_positions).cpu()

# Convert to NumPy if possible (or use directly in torch)
try:
    predicted_np = predicted_positions.numpy()
    true_np = true_positions.numpy()
except RuntimeError:
    import numpy as np
    predicted_np = np.array(predicted_positions.tolist())
    true_np = np.array(true_positions.tolist())

#save the results of predicted and true positions in a dataframe
import pandas as pd
df = pd.DataFrame({
    'true_X': true_np[:, 0],
    'true_Y': true_np[:, 1],
    'predicted_X': predicted_np[:, 0],
    'predicted_Y': predicted_np[:, 1]
})

save_dir = Path('results')
save_dir.mkdir(parents=True, exist_ok=True)
csv_path = save_dir / 'position_results.csv'
df.to_csv(csv_path, index=False)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(true_np[:, 0], true_np[:, 1], label='True', c='blue', alpha=0.6)
plt.scatter(predicted_np[:, 0], predicted_np[:, 1], label='Predicted', c='red', marker='x')
plt.title("Predicted vs Actual Object Positions")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
