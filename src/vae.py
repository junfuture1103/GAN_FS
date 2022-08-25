import random

import torch
from torch import nn
from torch.nn.functional import mse_loss
from .gan import GANGModel as VAEDModel

import src
from src import config
from src.datasets import PositiveDataset

def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class VAEEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calculate_mu = nn.Sequential(
            nn.Linear(src.models.x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, src.models.z_size),
        )
        self.calculate_log_variance = nn.Sequential(
            nn.Linear(src.models.x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, src.models.z_size),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        mu = self.calculate_mu(x)
        log_variance = self.calculate_log_variance(x)

        sigma = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(mu)
        z = epsilon * sigma + mu
        return z, mu, sigma

class VAE:

    def __init__(self):
        self.e = VAEEModel().to(config.device)
        self.d = VAEDModel().to(config.device)
        self.e.eval()
        self.d.eval()

    def fit(self):
        # self.logger.info('Started training')
        # self.logger.debug(f'Using device: {config.device}')
        
        e_optimizer = torch.optim.Adam(
            params=self.e.parameters(),
            lr=config.config_vae.e_lr,
        )
        
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.config_vae.d_lr,
        )

        x = PositiveDataset()[:][0].to(config.device)
        print("x in VAE", x)
        for _ in range(config.config_vae.epochs):
            # clear gradients
            self.e.zero_grad()
            self.d.zero_grad()
            # calculate z, mu and sigma
            z, mu, sigma = self.e(x)
            # calculate x_hat
            x_hat = self.d(z)
            # calculate loss
            divergence = - 0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
            loss = divergence + mse_loss(x_hat, x)
            # calculate gradients
            loss.backward()
            # optimize models
            e_optimizer.step()
            d_optimizer.step()

        self.e.eval()
        self.d.eval()
        return

    def generate_z(self, size: int = 1):
        # print("g_z samples : ", random.choices(PositiveDataset().samples, k=size))
        seeds = torch.stack(random.choices(PositiveDataset().samples, k=size)).to(config.device)
        print("seeds in g_z : ", seeds)
        return self.e(seeds)[0]
