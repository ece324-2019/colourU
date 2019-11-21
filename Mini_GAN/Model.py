import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''class Generator(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, hidden_size1, 7, stride=2, padding=1),
            nn.LeakyReLU(), nn.Conv2d(hidden_size1, hidden_size2, 5, stride=2, padding=1),
            nn.LeakyReLU(), nn.Conv2d(hidden_size2, 64, 3),
            nn.ConvTranspose2d(64, hidden_size2, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_size2, hidden_size1, 5, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(), nn.ConvTranspose2d(hidden_size1, 3, 7, stride=2, padding=1, output_padding=1), nn.Tanh())

    def forward(self, x):
        u = self.gen(x)
        u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)
        return u
'''


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.LeakyReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.LeakyReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, dilation=2, padding=2), nn.LeakyReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(), nn.BatchNorm2d(128),

            nn.Conv2d(128, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        u = self.gen(x)
        u = torch.cat((x[:, 0:1, :, :], u), dim=1)
        return u


class Discriminator(nn.Module):
    def __init__(self, input_size, kernelSize, kernelNum, hidden_size, output_size, convlayers=3, fclayers=2):
        super(Discriminator, self).__init__()
        self.conv = nn.ModuleList()
        fcIn = (input_size - (kernelSize - 1)) // 2
        self.conv.append(nn.Conv2d(3, kernelNum, kernelSize))
        if convlayers >= 2:
            for i in range(convlayers - 1):
                self.conv.append(nn.Conv2d(kernelNum, kernelNum, kernelSize))
                fcIn = (fcIn - (kernelSize - 1)) // 2
        self.fc = nn.ModuleList()
        self.fcInput = kernelNum * fcIn * fcIn
        if fclayers == 1:
            self.fc.append(nn.Linear(self.fcInput, output_size))
        elif fclayers == 2:
            self.fc.append(nn.Linear(self.fcInput, hidden_size))
            self.fc.append(nn.Linear(hidden_size, output_size))
        else:
            self.fc.append(nn.Linear(self.fcInput, hidden_size))
            for i in range(hidden_size - 2):
                self.fc.append(nn.Linear(hidden_size, hidden_size))
            self.fc.append(nn.Linear(hidden_size, output_size))  # output layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for i in range(len(self.conv)):
            x = self.pool(F.leaky_relu((self.conv[i](x))))
        x = x.view(-1, self.fcInput)
        for i in range(len(self.fc) - 1):
            x = F.leaky_relu(self.fc[i](x))
        x = torch.sigmoid(self.fc[i+1](x))
        return x

    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(  # like the Composition layer you built
                nn.Linear(56 * 56 * 3, 1024 * 2),
                nn.ReLU(), nn.Linear(1024 * 2, 512 * 2),
                nn.ReLU(), nn.Linear(512 * 2, 256),
                nn.ReLU(), nn.Linear(256, 128),
                nn.ReLU(), nn.Linear(128, 64)
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 512 * 2), nn.ReLU(),
                nn.Linear(512 * 2, 1024 * 2), nn.ReLU(),
                nn.Linear(1024 * 2, 56 * 56 * 3), nn.ReLU()
            )

        def forward(self, x):
            u = self.encoder(x)
            print(u.shape)
            u = self.decoder(u)
            print(u.shape)
            u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)
            return u


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

        u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)

        return u