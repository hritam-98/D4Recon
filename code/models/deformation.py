import torch
import torch.nn as nn

class DeformationNetwork(nn.Module):
    """
    Decodes HexPlane features into Gaussian deformations.
    Outputs: delta_mu, delta_scale, delta_quat, delta_opacity, delta_color
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Heads for different Gaussian attributes
        self.head_mu = nn.Linear(hidden_dim, 3)
        self.head_scale = nn.Linear(hidden_dim, 3)
        self.head_quat = nn.Linear(hidden_dim, 4)
        self.head_opacity = nn.Linear(hidden_dim, 1)
        self.head_color = nn.Linear(hidden_dim, 3) # Assuming SH DC offset
        
        # Zero initialization for stability
        for head in [self.head_mu, self.head_scale, self.head_quat, self.head_opacity, self.head_color]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h):
        x = self.net(h)
        return {
            'd_xyz': self.head_mu(x),
            'd_scale': self.head_scale(x),
            'd_rot': self.head_quat(x),
            'd_opacity': self.head_opacity(x),
            'd_color': self.head_color(x)
        }