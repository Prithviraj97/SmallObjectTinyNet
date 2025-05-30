# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import pandas as pd
# import os
# import numpy as np

# class DotDataset(Dataset):
#     def __init__(self, image_dir, label_file, transform=None):
#         self.image_dir = image_dir
#         self.labels = pd.read_csv(label_file)  # assumes columns: 'filename','x','y'
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         row = self.labels.iloc[idx]
#         img_path = os.path.join(self.image_dir, row['filename'])
#         image = Image.open(img_path).convert('L') 
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
#         label = torch.tensor([row['x'], row['y']], dtype=torch.float32)
#         return image, label


import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision import transforms

class DotDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        # assumes columns: 'frame','x','y'
        self.labels = pd.read_csv(label_file)  
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        # Construct filename from frame number
        img_name = f"frame_{int(row['frame']):04d}.png"
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and process image
        image = Image.open(img_path).convert('L')  # convert to grayscale
        width, height = image.size
        # if self.transform:
        #     image = self.transform(image)
        # else:
        #     image = torch.FloatTensor(np.asarray(image)).unsqueeze(0) / 255.0
        # image = self.transform(image)
        # # Get coordinates as label
        # label = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        image = image.float().reshape((1, height, width)) / 255.0

        label = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        return image, label