import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
from DataProcessing import *
from Model import *

from time import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.)
    '''elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1.)
        m.bias.data.fill_(0)
'''

def train():
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/val/'
    # Settings and Hyperparameters
    fresh_start = False  # True to start new model False to continue training saved model
    d_import_name = 'good_d.pt'
    g_import_name = 'baseline.pt'
    d_export_name = 'good_d.pt'
    g_export_name = 'good_g.pt'
    g_error_scaler = 0.5
    g_train_scaler = 1
    num_images = 1000
    num_epochs = 50
    print_interval = 1
    batch_size = 64

    # Model Parameters
    g_hidden_size1 = 32
    g_hidden_size2 = 16
    d_input_size = 64    # dimension of square image
    d_kernel_size = 3    # kernel size for discriminator kernels
    d_kernel_number = 20  #number of kernels
    d_hidden_size = 16    # hidden MLP Layers
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    d_conv_layers = 1  # Number of convolutional layers
    d_fclayers = 2  # Number of fully connected layers
    d_learning_rate = 1e-4
    g_learning_rate = 1e-4
    sgd_momentum = 0.9

    d_steps = 1
    g_steps = g_train_scaler

    if fresh_start:
        G = Generator()
        G.float()
        D = Discriminator(input_size=d_input_size, kernelSize=d_kernel_size, kernelNum=d_kernel_number,
                      hidden_size=d_hidden_size, output_size=d_output_size, convlayers=d_conv_layers,
                      fclayers=d_fclayers)
        D.float()
        D.apply(weights_init)
        G.apply(weights_init)
    else:
        try:
            '''G = Autoencoder()
            G.load_state_dict(torch.load(g_import_name))
            G.eval()'''
            G = torch.load(g_import_name)
            print('imported')
            D = Discriminator(input_size=d_input_size, kernelSize=d_kernel_size, kernelNum=d_kernel_number,
                              hidden_size=d_hidden_size, output_size=d_output_size, convlayers=d_conv_layers,
                              fclayers=d_fclayers)
            D.float()
            #D = torch.load(d_import_name)
        except:
            G = Generator()
            G.float()
            D = Discriminator(input_size=d_input_size, kernelSize=d_kernel_size, kernelNum=d_kernel_number,
                              hidden_size=d_hidden_size, output_size=d_output_size, convlayers=d_conv_layers,
                              fclayers=d_fclayers)
            D.float()

    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    criterion2 = nn.MSELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)  #, momentum=sgd_momentum)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)  #, momentum=sgd_momentum)

    d_loss = np.zeros(num_epochs*d_steps)
    g_loss = np.zeros(num_epochs*g_steps)

    d_loss_index = 0
    g_loss_index = 0

    # Import Data
    img_tensor = import_folder(image_path, num_imgs=num_images).float()
    gray_img_tensor = process(img_tensor)
    DT = TensorDataset(gray_img_tensor, img_tensor)
    train_loader = DataLoader(DT, batch_size=batch_size, shuffle=True)

    t_init = time()
    for epoch in range(num_epochs):
        D.train()
        G.eval()
        for data in train_loader:
            gray, real = data
            D.zero_grad()

            # Train discriminator on real
            d_real_decision = D(real)
            d_real_error = criterion(d_real_decision.squeeze(), torch.zeros([d_real_decision.shape[0]]))
            d_real_error.backward()

            # Train on the fake
            d_fake_data = G(gray).detach()
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision.squeeze(), torch.ones([d_fake_decision.shape[0]]))
            d_fake_error.backward()

            d_optimizer.step()

        d_loss[d_loss_index] = (d_real_error + d_fake_error) / (d_real_decision.shape[0] + d_fake_decision.shape[0])
        d_loss_index += 1

        D.eval()
        G.train()

        for i in range(g_steps):
            for data in train_loader:
                gray, real = data

                G.zero_grad()

                g_fake_data = G(gray)
                dg_fake_decision = D(g_fake_data)
                g_error_1 = criterion(dg_fake_decision.squeeze(), torch.zeros([dg_fake_decision.shape[0]]))  # Train G to pretend it's genuine
                g_error_2 = criterion2(g_fake_data, real)
                g_error = g_error_1 + g_error_scaler * g_error_2
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters'''

            g_loss[g_loss_index] = g_error / dg_fake_decision.shape[0]
            g_loss_index += 1

        if epoch % print_interval == 0:
            print("(", time() - t_init, ") Epoch", epoch, ": D (error:", d_loss[d_loss_index - 1], ") G (error:", g_loss[g_loss_index - 1], "); ")

    torch.save(D, d_export_name)
    torch.save(G, g_export_name)

    '''if g_steps==1:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('GAN Losses')
        fig1 = fig.add_subplot(211)
        ax1.plot(np.arange(0, d_loss_index), d_loss, label='Discriminator')
        ax2.plot(np.arange(0, g_loss_index), g_loss, label='Generator')
        fig.set_ylabel('Loss')
        fig.set_xlabel('Epoch')
        fig.legend()
    else:'''
    np.save('loss_g.txt', g_loss)
    np.save('loss_d.txt', d_loss)
    plt.plot(np.arange(0, d_loss_index), d_loss, label='Discriminator')
    plt.legend()
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(np.arange(0, g_loss_index), g_loss, label='Generator')
    plt.legend()
    plt.title('GAN Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    gen_input = import_folder(image_path, num_imgs=5, shuffle=True).float()
    g_fake_data = G(process(gen_input)).detach()
    display_processed_imgs(gen_input.detach(), g_fake_data.detach(), 4)


train()
