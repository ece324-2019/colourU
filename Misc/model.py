import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),nn.Conv2d(32, 64, 7))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        u = self.encoder(x)
        u = self.decoder(u)

        u = torch.stack((x[:,0,:,:], u[:,0,:,:], u[:,1,:,:]), dim=1)

        return u

class gen_paper(nn.Module):
    def __init__(self):
        super(gen_paper, self).__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),                          # Block 1
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),              # Block 2
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(),             # Block 3
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, padding=1, stride=2), nn.ReLU(),             # Block 4
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),           # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),           # Block 6
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),                       # Block 7
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),    # Block 8
            nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),    # Block 9
            nn.ConvTranspose2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),    # Block 10
            nn.ConvTranspose2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 2, 1)
        )

    def forward(self, x):
        u = self.generator(x)
        u = torch.stack((x[:,0,:,:], u[:,0,:,:], u[:,1,:,:]), dim=1)
        return u

class better_generator(nn.Module):
    def __init__(self):
        super(better_generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(), nn.BatchNorm2d(128),

            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x):
        u = torch.clamp(self.gen(x), -127, 127)
        u = torch.cat((x[:,0:1,:,:], u), dim=1)
        return u