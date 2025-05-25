import os
import csv
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_synthetic_dataset(
    save_dir="synthetic_dataset",
    num_frames=1000,
    image_size=(96, 96),
    dot_radius=2.5,
    initial_pos=(10, 10),
    velocity=(1, 1),
    jitter=0.5,
    seed=42):

    np.random.seed(seed)
    random.seed(seed)

    # Create necessary folders
    image_dir = os.path.join(save_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Labels CSV for positions of the dot in each frame
    labels_path = os.path.join(save_dir, "labels.csv")
    with open(labels_path, mode='w', newline='') as csvfile:
        fieldnames = ['frame', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Initialize position and velocity of the dot
        x, y = initial_pos
        vx, vy = velocity

        for i in range(num_frames):
            # Create blank image
            img = Image.new("L", image_size, color=0)
            draw = ImageDraw.Draw(img)

            # Draw the dot
            draw.ellipse(
                (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                fill=255
            )
            # Save image and Save label
            img.save(os.path.join(image_dir, f"frame_{i:04d}.png"))
            writer.writerow({'frame': f'{i:04d}', 'x': int(x), 'y': int(y)})

            # Update position with velocity and jitter
            x += vx + np.random.uniform(-jitter, jitter)
            y += vy + np.random.uniform(-jitter, jitter)

            # Keep position within bounds
            x = np.clip(x, dot_radius, image_size[0] - dot_radius)
            y = np.clip(y, dot_radius, image_size[1] - dot_radius)

    print(f"Synthetic dataset created at: {save_dir}")

# Generate the dataset
if __name__ == "__main__":
    generate_synthetic_dataset()
