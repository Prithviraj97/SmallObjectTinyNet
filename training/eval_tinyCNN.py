import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from dataset import DotDataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.tinyCNN import TinyCNNRegressor

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "tinycnn_trained.pth"

# ----------------------------
# Load Test Dataset
# ----------------------------
transform = transforms.ToTensor()
test_dataset = DotDataset(
    image_dir="synthetic_dataset/test/images",
    label_file="synthetic_dataset/test/labels.csv",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ----------------------------
# Load Trained Model
# ----------------------------
model = TinyCNNRegressor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# Inference and Collect Data
# ----------------------------
predicted_x, predicted_y = [], []
true_x, true_y = [], []
records = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images).cpu().squeeze()
        labels = labels.squeeze()

        outputs_list = outputs.tolist()
        labels_list = labels.tolist()

        predicted_x.append(outputs_list[0])
        predicted_y.append(outputs_list[1])
        true_x.append(labels_list[0])
        true_y.append(labels_list[1])

        records.append({
            'pred_x': outputs_list[0],
            'pred_y': outputs_list[1],
            'true_x': labels_list[0],
            'true_y': labels_list[1]
        })

# ----------------------------
# Create DataFrame
# ----------------------------
df = pd.DataFrame(records)
df.to_csv("tinycnn_predictions.csv", index=False)
print("Saved predictions to tinycnn_predictions.csv")

# ----------------------------
# 2D Scatter Plot
# ----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(true_x, true_y, color='green', label='Ground Truth', s=20)
plt.scatter(predicted_x, predicted_y, color='red', label='Predicted', s=20, marker='x')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('TinyCNN Predictions vs Ground Truth (Test Set)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("tinycnn_2d_scatter.png", dpi=300)
plt.show()
