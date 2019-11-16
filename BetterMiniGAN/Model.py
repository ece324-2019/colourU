import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 2, 1)
        )

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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.)
        m.bias.data.fill_(0)