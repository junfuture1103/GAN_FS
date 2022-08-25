import torch
from torch import nn

import src

def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class JUNGANDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_1 = nn.Sequential(
            nn.Linear(src.models.x_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(0.2),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.ReLU(0.2),
        )
        self.step_2 = nn.Linear(8, 1)

        self.hidden_output = None

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        self.hidden_output = self.step_1(x)
        output = self.step_2(self.hidden_output)
        return output
