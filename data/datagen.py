'''
This script generates a sequence of frames with a moving dot and saves the frames and positions.
This is synthetic data for testing the model.
The dot moves in a 2D space with some noise added to its trajectory.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_sequence(num_frames=10, img_size=64, dot_radius=2, noise_std=0.5):
    x, y = np.random.uniform(10, img_size-10, size=2)
    vx, vy = np.random.uniform(-2, 2, size=2)

    frames = []
    positions = []

    for _ in range(num_frames):
        img = np.zeros((img_size, img_size), dtype=np.float32)

        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), dot_radius, 1.0, -1)

        frames.append(img)
        positions.append((x, y))

        x += vx + np.random.normal(0, noise_std)
        y += vy + np.random.normal(0, noise_std)

        x = np.clip(x, dot_radius, img_size - dot_radius)
        y = np.clip(y, dot_radius, img_size - dot_radius)

    return np.stack(frames), np.array(positions)

frames, positions = generate_sequence(num_frames=20)
#visualize the generated frames and positions
for i in range(len(frames)):
    plt.imshow(frames[i], cmap='gray')
    plt.title(f'Frame {i} | Position: ({positions[i][0]:.2f}, {positions[i][1]:.2f})')
    plt.axis('off')
    plt.show()
