import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(32, 64, 7),
            nn.ReLU(), nn.Conv2d(64, 128, 3)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        u = self.encoder(x)
        u = self.decoder(u)
        u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)
        return u

def train(model, data, num_epochs=5,  batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # <--
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loss_array = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for data in train_loader:
            img_in, img_out = data
            recon = model(img_in)
            loss = criterion(recon, img_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_array[epoch] += loss

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))

    plt.plot(np.arange(0, num_epochs), loss_array, label='Model')
    plt.legend()
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model