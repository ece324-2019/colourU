import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, kernel_num, kernel_size):  # kernel_num original = 16, 32, 64, 128 kernel_size: 3,3,7,3
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, kernel_num[0], kernel_size[0], stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(kernel_num[0], kernel_num[1], kernel_size[1], stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(kernel_num[1], kernel_num[2], kernel_size[2]),
            nn.ReLU(), nn.Conv2d(kernel_num[2], kernel_num[3], kernel_size[3])
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kernel_num[3], kernel_num[2], kernel_size[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_num[2], kernel_num[1], kernel_size[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_num[1], kernel_num[0], kernel_size[1], stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.ConvTranspose2d(kernel_num[0], 3, kernel_size[0], stride=2, padding=1, output_padding=1))

    def forward(self, x):
        u = self.encoder(x)
        u = self.decoder(u)
        u = torch.stack((x[:, 0, :, :], u[:, 0, :, :], u[:, 1, :, :]), dim=1)
        return u


def train(model, data, val_data, num_imgs, num_val_imgs, num_epochs=5,  batch_size=64, learning_rate=1e-3, name='null'):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # <--
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    loss_array = np.zeros(num_epochs)
    val_loss_array = np.zeros(num_epochs)
    min_val_loss = 1000000
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            img_in, img_out = data
            recon = model(img_in)
            loss = criterion(recon, img_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_array[epoch] += loss
        loss_array[epoch] = loss_array[epoch]/num_imgs
        model.eval()
        for data in val_loader:
            img_in, img_out = data
            recon = model(img_in)
            loss = criterion(recon, img_out)
            val_loss_array[epoch] += loss  #acummulate
        val_loss_array[epoch] = val_loss_array[epoch] / num_val_imgs  # normalize
        if val_loss_array[epoch] < min_val_loss:
            torch.save(model, 'best'+name+'.pt')
        print('Epoch:{}, Train Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch+1, float(loss_array[epoch]),
                                                                    float(val_loss_array[epoch])))

    plt.plot(np.arange(0, num_epochs), loss_array, label='training')
    plt.plot(np.arange(0, num_epochs), val_loss_array, label='validation')
    plt.legend()
    plt.title('Losses:'+name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    return model, min_val_loss