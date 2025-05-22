'''
testing the effect of implementing the Kullback-Leibler divergence loss function
on the model's performance
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Dummy FusionNet-like module to simulate lambda(t) evolution
class DummyFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_logit = nn.Parameter(torch.tensor(0.0))  # Start at 0.5 (sigmoid(0) = 0.5)

    def forward(self):
        return torch.sigmoid(self.lambda_logit)  # λ(t)

# KL divergence between λ(t) and λ0
def kl_divergence_lambda(lambda_t, lambda_0=0.5):
    lambda_0 = torch.tensor(lambda_0)
    return lambda_t * torch.log(lambda_t / lambda_0) + (1 - lambda_t) * torch.log((1 - lambda_t) / (1 - lambda_0))

# Simulate training process over time for different beta values
def simulate_lambda_evolution(betas, timesteps=30):
    lambda_records = {beta: [] for beta in betas}

    for beta in betas:
        model = DummyFusionNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for t in range(timesteps):
            lambda_t = model()
            # Simulated fusion loss (assume it's always minimized at λ=0.8)
            fusion_loss = (lambda_t - 0.8) ** 2

            # Total loss with KL regularization
            kl = kl_divergence_lambda(lambda_t)
            total_loss = fusion_loss + beta * kl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            lambda_records[beta].append(lambda_t.item())

    return lambda_records

# Simulate and plot
betas = [0.0, 0.1, 0.5, 1.0]
lambda_records = simulate_lambda_evolution(betas)

# Plotting
plt.figure(figsize=(10, 6))
for beta, lambdas in lambda_records.items():
    plt.plot(lambdas, label=f"β = {beta}")
plt.title("Evolution of λ(t) over time for different β (KL Regularization)")
plt.xlabel("Time step (frame)")
plt.ylabel("λ(t) value")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
