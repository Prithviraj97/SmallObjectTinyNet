import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SyntheticDotDataset(Dataset):
    def __init__(self, num_sequences=1000, frames_per_seq=10, img_size=64,
                 dot_radius=2, noise_std=0.5, transform=None):
        self.num_sequences = num_sequences
        self.frames_per_seq = frames_per_seq
        self.img_size = img_size
        self.dot_radius = dot_radius
        self.noise_std = noise_std
        self.transform = transform

        # Generate all sequences at init
        self.frames, self.positions = self._generate_dataset()

        # Flatten so each (frame, position) is an individual sample
        self.data = []
        for seq_frames, seq_positions in zip(self.frames, self.positions):
            for frame, pos in zip(seq_frames, seq_positions):
                self.data.append((frame, pos))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, pos = self.data[idx]
        frame = np.expand_dims(frame, axis=0)  # [1, H, W] for grayscale
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        label_tensor = torch.tensor(pos, dtype=torch.float32)
        if self.transform:
            frame_tensor = self.transform(frame_tensor)
        return frame_tensor, label_tensor

    def _generate_sequence(self):
        x, y = np.random.uniform(10, self.img_size-10, size=2)
        vx, vy = np.random.uniform(-2, 2, size=2)

        frames = []
        positions = []

        for _ in range(self.frames_per_seq):
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(img, (cx, cy), self.dot_radius, 1.0, -1)

            frames.append(img)
            positions.append((x, y))

            x += vx + np.random.normal(0, self.noise_std)
            y += vy + np.random.normal(0, self.noise_std)
            x = np.clip(x, self.dot_radius, self.img_size - self.dot_radius)
            y = np.clip(y, self.dot_radius, self.img_size - self.dot_radius)

        return np.stack(frames), np.array(positions)

    def _generate_dataset(self):
        all_frames = []
        all_positions = []
        for _ in range(self.num_sequences):
            frames, positions = self._generate_sequence()
            all_frames.append(frames)
            all_positions.append(positions)
        return np.array(all_frames), np.array(all_positions)
