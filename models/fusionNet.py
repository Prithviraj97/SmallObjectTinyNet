import torch
import torch.nn as nn
import torch.nn.functional as F

# Physics-based predictor: simple constant velocity model
class PhysicsModule(nn.Module):
    def forward(self, x_t_minus_1, x_t_minus_2):
        return x_t_minus_1 + (x_t_minus_1 - x_t_minus_2)

# Tiny CNN-based predictor
class TinyCNN(nn.Module):
    # def __init__(self):
    #     super(TinyCNN, self).__init__()
    #     self.cnn = nn.Sequential(
    #         nn.Conv2d(1, 8, 3, padding=1),  # (B, 1, 64, 64) -> (B, 8, 64, 64)
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),               # -> (B, 8, 32, 32)
    #         nn.Conv2d(8, 16, 3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),               # -> (B, 16, 16, 16)
    #         nn.Flatten(),                  # -> (B, 4096)
    #         nn.Linear(16 * 16 * 16, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 2)               # Output: (x, y)
    #     )

    # def forward(self, x):
    #     return self.cnn(x)
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # output x, y position
        )

    def forward(self, x):
        return self.fc(self.cnn(x))

# Learnable Fusion Network
class FusionNet(nn.Module):
    def __init__(self, use_kl=True, lambda_0=0.4):
        super(FusionNet, self).__init__()
        self.cnn_model = TinyCNN()
        self.physics = PhysicsModule()
        self.lambda_t = nn.Parameter(torch.tensor(0.5))  # Learnable scalar
        self.use_kl = use_kl
        self.lambda_0 = lambda_0

    def forward(self, x_t, x_t_minus_1_pos, x_t_minus_2_pos):
        x_cnn = self.cnn_model(x_t)
        x_phys = self.physics(x_t_minus_1_pos, x_t_minus_2_pos)
        lambda_t_clipped = torch.clamp(self.lambda_t, 0.0, 1.0)
        x_fused = lambda_t_clipped * x_cnn + (1 - lambda_t_clipped) * x_phys
        return x_fused, x_cnn, x_phys, lambda_t_clipped

# Custom loss with KL regularization
def fusion_loss(pred, target, lambda_t, lambda_0=0.5, beta=0.1, use_kl=True):
    mse = F.mse_loss(pred, target)

    if use_kl:
        eps = 1e-6
        lambda_clipped = torch.clamp(lambda_t, eps, 1 - eps)
        kl_div = lambda_clipped * torch.log(lambda_clipped / lambda_0) + \
                 (1 - lambda_clipped) * torch.log((1 - lambda_clipped) / (1 - lambda_0))
        return mse + beta * kl_div
    else:
        return mse

