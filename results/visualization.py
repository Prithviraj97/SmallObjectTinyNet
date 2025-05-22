import torch
import matplotlib.pyplot as plt

# Load saved loss and lambda histories
loss_history = torch.load("fusion_loss_history.pt")
lambda_history = torch.load("fusion_lambda_history.pt")

# Plot training loss over epochs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, marker='o', label='Fusion Loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Plot lambda_t over epochs
plt.subplot(1, 2, 2)
plt.plot(lambda_history, marker='x', color='orange', label='λ(t)')
plt.title("Fusion Weight λ(t) Over Time")
plt.xlabel("Epoch")
plt.ylabel("Lambda Value")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("fusionnet_training_diagnostics.png")
plt.show()
