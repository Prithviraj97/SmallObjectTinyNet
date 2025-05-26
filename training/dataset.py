import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class DotDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_file)  # assumes columns: 'filename','x','y'
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        return image, label
