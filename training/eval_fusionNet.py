import torch
import pandas as pd
from collections import deque
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fusionNet import FusionNet
from training.dataset import DotDataset

# ---------------------------
# Configuration
# ---------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
MODEL_PATH = 'fusionNet_model.pth' 
OUTPUT_CSV = 'fusionnet_predictions.csv'

# ---------------------------
# Transforms and Dataset
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_loader(split='test'):
    dataset = DotDataset(
        image_dir=f'synthetic_dataset/{split}/images',
        label_file=f'synthetic_dataset/{split}/labels.csv',
        transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

test_loader = get_loader('test')

# ---------------------------
# Load Model
# ---------------------------
model = FusionNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------------------
# Evaluation
# ---------------------------
results = []
pos_queue = deque(maxlen=3)

with torch.no_grad():
    for idx, (img, label) in enumerate(test_loader):
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        pos_queue.append(label)

        if len(pos_queue) < 3:
            continue

        pos1, pos2, target = pos_queue
        pred, _, _, _ = model(img, pos1, pos2)

        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        results.append({
            'image_index': idx,
            'x_pred': pred_np[0],
            'y_pred': pred_np[1],
            'x_true': target_np[0],
            'y_true': target_np[1],
        })

# ---------------------------
# Save CSV
# ---------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved evaluation results to {OUTPUT_CSV}")
