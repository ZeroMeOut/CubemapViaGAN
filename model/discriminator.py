import torch 
import torch.nn as nn

## A simple discriminator
## I am probably ditch this
class discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)