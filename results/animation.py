import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic_dataset import SyntheticDotDataset
from models.fusionNet import FusionNet, fusion_loss
# Assuming these exist already:
# - FusionNet
# - SyntheticTrackingDataset

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

    # def __getitem__(self, idx):
    #     img, pos1, pos2, target = self.samples[idx]
    #     return torch.tensor(img).unsqueeze(0), torch.tensor(pos1), torch.tensor(pos2), torch.tensor(target)

    def __getitem__(self, idx):
        img, pos1, pos2, target = self.samples[idx]
        return (
            torch.tensor(img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(pos1, dtype=torch.float32),
            torch.tensor(pos2, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )
    
# Load model
model = FusionNet()
model.load_state_dict(torch.load("fusion_model_small.pth"))
model.eval()

# Generate or load test data
test_dataset = SyntheticTrackingDataset(num_sequences=1, seq_len=30)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Get the first sequence (only 1 in test set)
img_seq, pos1, pos2, target = next(iter(test_loader))
with torch.no_grad():
    pred, _, _, _ = model(img_seq, pos1, pos2)
    true_positions = target.cpu().detach().tolist()
    pred_positions = pred.cpu().detach().tolist()

# Convert lists to numpy arrays
true_positions = np.array(true_positions).squeeze()
pred_positions = np.array(pred_positions).squeeze()

# Convert tensors to numpy
    # true_positions = np.array(target.tolist())
    # pred_positions = np.array(pred.tolist())

# Animation setup
# fig, ax = plt.subplots()
# ax.set_xlim(0, 64)
# ax.set_ylim(0, 64)
# ax.set_title("Tracking: Predicted vs Ground Truth")

# true_dot, = ax.plot([], [], 'go', label="Ground Truth")  # green dot
# pred_dot, = ax.plot([], [], 'ro', label="Prediction")    # red dot
# ax.legend()

# def update(frame):
#     x_true, y_true = true_positions[frame]
#     x_pred, y_pred = pred_positions[frame]
#     true_dot.set_data(x_true, y_true)
#     pred_dot.set_data(x_pred, y_pred)
#     return true_dot, pred_dot

# ani = animation.FuncAnimation(fig, update, frames=len(true_positions), blit=True, interval=200)

# # Save or show animation
# ani.save("tracking_animation.gif", writer='pillow')  # Save as GIF
# plt.show()

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 64)
ax.set_ylim(0, 64)
ax.set_title("Tracking: Predicted vs Ground Truth")
ax.grid(True)

# Initialize empty plots
true_dot, = ax.plot([], [], 'go-', label="Ground Truth", markersize=10)
pred_dot, = ax.plot([], [], 'ro--', label="Prediction", markersize=10)
true_point, = ax.plot([], [], 'go', markersize=10, label="Current True Position")
pred_point, = ax.plot([], [], 'ro', markersize=10, label="Current Predicted Position")
ax.legend()

# Initialize function (required for blitting)
def init():
    true_dot.set_data([], [])
    pred_dot.set_data([], [])
    true_point.set_data([], [])
    pred_point.set_data([], [])
    return true_dot, pred_dot, true_point, pred_point

# def update(frame):
#     x_true, y_true = true_positions[frame]
#     x_pred, y_pred = pred_positions[frame]
#     true_dot.set_data([x_true], [y_true])
#     pred_dot.set_data([x_pred], [y_pred])
#     return true_dot, pred_dot

def animate(frame):
    # Plot trajectory up to current frame
    true_dot.set_data(true_positions[:frame+1, 0], true_positions[:frame+1, 1])
    pred_dot.set_data(pred_positions[:frame+1, 0], pred_positions[:frame+1, 1])
    # Update current positions
    true_point.set_data([true_positions[frame, 0]], [true_positions[frame, 1]])
    pred_point.set_data([pred_positions[frame, 0]], [pred_positions[frame, 1]])
    return true_dot, pred_dot, true_point, pred_point

# def update(frame):
#     ax.clear()
#     ax.set_xlim(0, 64)
#     ax.set_ylim(0, 64)
#     ax.set_title(f"Frame {frame+1}")
#     ax.plot(*true_positions[frame], 'go', label='Ground Truth')  # green
#     ax.plot(*pred_positions[frame], 'ro', label='Prediction')    # red
#     ax.legend(loc='upper right')
#     return ax

def update(frame):
    ax.clear()
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.grid(True)
    
    # Plot full trajectories with opacity
    ax.plot(true_positions[:frame+1, 0], true_positions[:frame+1, 1], 
            'g-', alpha=0.5, label='Ground Truth Path')
    ax.plot(pred_positions[:frame+1, 0], pred_positions[:frame+1, 1], 
            'r--', alpha=0.5, label='Predicted Path')
    
    # Plot current positions
    ax.plot(true_positions[frame, 0], true_positions[frame, 1], 
            'go', markersize=10, label='Current True')
    ax.plot(pred_positions[frame, 0], pred_positions[frame, 1], 
            'ro', markersize=10, label='Current Predicted')
    
    ax.set_title(f'Frame {frame+1}/{len(true_positions)}')
    ax.legend(loc='upper right')

# Create animation with init_func
ani = animation.FuncAnimation(
    fig, 
    update,
    
    frames=len(true_positions),
    interval=200,
    blit=False,
    repeat=True
)

# Save animation
# plt.rcParams['animation.writer'] = 'pillow'
ani.save("tracking_animation2.gif", writer='pillow', fps=2,
         progress_callback=lambda i, n: print(f'Saving frame {i+1}/{n}'))
plt.close()