import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from time import time

class GenResNet(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3):
        super(GenResNet, self).__init__()

        self.c1 = nn.Conv2d(3, hidden_size1, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(hidden_size1, hidden_size2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(hidden_size2, hidden_size3, 3)
        self.c4 = nn.Conv2d(hidden_size3, 64, 3)
        self.ct4 = nn.ConvTranspose2d(64, hidden_size3, 3)
        self.ct3 = nn.ConvTranspose2d (hidden_size3, hidden_size2, 3)
        self.ct2 = nn.ConvTranspose2d (hidden_size2, hidden_size1, 3, stride=2, padding=1, output_padding=1)
        self.ct1 = nn.ConvTranspose2d (hidden_size1, 2, 3, stride=2, padding=1, output_padding=1)

        self.res3 = nn.Conv2d(hidden_size3, hidden_size3, 3, padding=1)
        self.res2 = nn.Conv2d(hidden_size2, hidden_size2, 3, padding=1)
        self.res1 = nn.Conv2d(hidden_size1, hidden_size1, 3, padding=1)

    def forward(self, inputs):
        next = F.leaky_relu(self.c1(inputs))
        res1 = self.res1(next)
        next = F.leaky_relu(self.c2(next))
        res2 = self.res2(next)
        next = F.leaky_relu(self.c3(next))
        res3 = self.res3(next)
        next = F.leaky_relu(self.c4(next))
        next = F.leaky_relu(self.ct4(next) + res3)
        next = F.leaky_relu(self.ct3(next) + res2)
        next = F.leaky_relu(self.ct2(next) + res1)
        next = self.ct1(next)

        return torch.stack((inputs[:,0,:,:], next[:,0,:,:], next[:,1,:,:]), dim=1)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),nn.Conv2d(32, 64, 3))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        u = self.encoder(x)
        u = self.decoder(u)

        u = torch.stack((x[:,0,:,:], u[:,0,:,:], u[:,1,:,:]), dim=1)

        return u

class Generator(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, hidden_size1, 3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(hidden_size1, hidden_size2, 3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(hidden_size2, 64, 3),
            nn.ConvTranspose2d(64, hidden_size2, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size2, hidden_size1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.ConvTranspose2d(hidden_size1, 2, 3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        u = self.gen(x)
        u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)
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
            x = self.pool(F.relu(self.conv[i](x)))
        x = x.view(-1, self.fcInput)
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = torch.sigmoid(self.fc[i+1](x))
        return x