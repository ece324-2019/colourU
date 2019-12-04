import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from time import time
from DataProcessing import *

from matplotlib import pyplot as plt

import numpy as np

def d_accuracy(array, isFake):
    acc = 0
    for i in range(len(array)):
        if (isFake and array[i] > 0.5) or (not isFake and array[i] <= 0.5):
            acc += 1
    return acc


def train_GAN(G, D, train_loader, val_loader, test_loader, train_num, val_num, test_num, source_path, pretraining=True,
              num_epochs=5, out_file=None, d_learning_rate=1e-4, g_learning_rate=2e-4, m_param='N/A'):
    # Settings and Hyperparameters
    g_error_scaler = 0.005
    g_train_scaler = 20
    g_pretrain_epoch = 10
    d_pretrain_epoch = 10
    print_interval = 1
    val_interval = 5
    sgd_momentum = 0.5

    # Model Parameters
    d_steps = 1
    g_steps = g_train_scaler

    # load images
    criterion = nn.BCELoss()
    criterion2 = nn.MSELoss()
    loss_normalizer = nn.MSELoss()
    d_optimizer_pretrain = optim.Adam(D.parameters(), lr=d_learning_rate, betas=(sgd_momentum, 0.999))
    g_optimizer_pretrain = optim.Adam(G.parameters(), lr=g_learning_rate, betas=(sgd_momentum, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=(sgd_momentum, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=(sgd_momentum, 0.999))

    d_loss = np.zeros(num_epochs)
    g_loss = np.zeros(num_epochs)
    g_val_loss = np.zeros((-(-num_epochs // val_interval)) - 1)
    d_val_accuracy = np.zeros((-(-num_epochs // val_interval)) - 1)

    d_loss_index = 0
    g_loss_index = 0
    g_val_loss_index = 0

    min_val_loss = 1000000
    min_val_loss_epoch = 0
    # Pretraining Generator
    if pretraining:
        print("Pretrain Generator")
        for epoch in range(g_pretrain_epoch):
            print('Epoch:', epoch)
            G.train()
            for data in train_loader:
                gray, real = data
                g_optimizer_pretrain.zero_grad()
                g_fake_data = G(gray)
                g_loss_train = criterion2(g_fake_data, real)
                g_loss_train.backward()
                g_optimizer_pretrain.step()

        # Pre-training Discriminator
        print("Pre-train Discriminator")
        for epoch in range(d_pretrain_epoch):
            print('Epoch:', epoch)
            G.eval()
            D.train()
            for data in train_loader:
                gray, real = data
                d_optimizer_pretrain.zero_grad()

                # Train discriminator on real
                d_real_decision = D(real)
                d_real_error = criterion(d_real_decision.squeeze(), torch.zeros([d_real_decision.shape[0]]))
                d_real_error.backward()

                # Train on the fake
                d_fake_data = G(gray).detach()
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision.squeeze(), torch.ones([d_fake_decision.shape[0]]))
                d_fake_error.backward()

                d_optimizer_pretrain.step()

    # Train GAN
    print('Train GAN')
    t_init = time()
    val_x_axis = []
    for epoch in range(num_epochs):
        D.train()
        G.eval()

        for data in train_loader:
            gray, real = data
            d_optimizer.zero_grad()

            # Train discriminator on real images
            d_real_decision = D(real)
            d_real_error = criterion(d_real_decision.squeeze(), torch.zeros([d_real_decision.shape[0]]))
            d_real_error.backward()

            # Train on the fake images
            d_fake_data = G(gray).detach()
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision.squeeze(), torch.ones([d_fake_decision.shape[0]]))
            d_fake_error.backward()

            d_loss[d_loss_index] += d_real_error + d_fake_error

            d_optimizer.step()

        d_loss[d_loss_index] = d_loss[d_loss_index] / (2 * train_num)
        d_loss_index += 1

        D.eval()
        G.train()

        for i in range(g_steps):
            for data in train_loader:
                gray, real = data
                g_optimizer.zero_grad()

                g_fake_data = G(gray)
                dg_fake_decision = D(g_fake_data)
                g_error_1 = criterion(dg_fake_decision.squeeze(),
                                      torch.zeros([dg_fake_decision.shape[0]]))  # Train G to pretend it's genuine
                g_error_2 = criterion2(g_fake_data, real)
                g_error = g_error_1 + g_error_scaler * g_error_2
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

                g_loss[g_loss_index] += g_error

        g_loss[g_loss_index] = g_loss[g_loss_index] / (g_steps * train_num)
        g_loss_index += 1

        if epoch % val_interval == 0 and epoch != 0:
            G.eval()
            D.eval()
            for data in val_loader:
                gray, real = data
                g_fake_data = G(gray)
                g_error = criterion2(g_fake_data, real)
                normalization = loss_normalizer(gray[:, :3, :, :], real)
                g_val_loss[g_val_loss_index] += g_error / normalization
                d_decision = D(g_fake_data)
                # print(d_decision)
                d_val_accuracy[g_val_loss_index] += d_accuracy(d_decision, True)
                d_decision = D(real)
                # print(d_decision)
                d_val_accuracy[g_val_loss_index] += d_accuracy(d_decision, False)

            d_val_accuracy[g_val_loss_index] = d_val_accuracy[g_val_loss_index] / (2 * val_num)
            g_val_loss[g_val_loss_index] = g_val_loss[g_val_loss_index] / val_num
            val_x_axis.append(epoch)
            print("Validation Loss: ", g_val_loss[g_val_loss_index], "Discriminator Accuracy: ",
                  d_val_accuracy[g_val_loss_index])

            if g_val_loss[g_val_loss_index] < min_val_loss:
                torch.save(D, source_path + out_file + "_interm_D.pt")
                torch.save(G, source_path + out_file + "_interm_G.pt")
                torch.save(D.state_dict(), source_path + out_file + "_state_dict_interm_D.pt")
                torch.save(G.state_dict(), source_path + out_file + "_state_dict_interm_G.pt")
                min_val_loss = g_val_loss[g_val_loss_index]
                min_val_loss_epoch = epoch
                print("Saved at epoch:", epoch)
            g_val_loss_index += 1

        if epoch % print_interval == 0:
            print("(", time() - t_init, ") Epoch", epoch, ": D (error:", d_loss[d_loss_index - 1], ") G (error:",
                  g_loss[g_loss_index - 1], "); ")

    torch.save(D, source_path + out_file + "_D.pt")
    torch.save(G, source_path + out_file + "_G.pt")
    torch.save(D.state_dict(), source_path + out_file + "_state_dict_D.pt")
    torch.save(G.state_dict(), source_path + out_file + "_state_dict_G.pt")

    # Validation Discriminator Accuracy Plot
    plt.plot(val_x_axis, d_val_accuracy, label='Discriminator')
    plt.legend()
    plt.title('GAN: Discriminator Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(source_path + out_file + "_Val_Discriminator_Accuracy.png", )
    plt.show()

    # Validation Loss Plot
    plt.plot(val_x_axis, g_val_loss, label='Generator')
    plt.legend()
    plt.title('GAN: Normalized Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Normalized Loss')
    plt.savefig(source_path + out_file + "_Normalized_Validation_GAN_Loss.png", )
    plt.show()

    # GAN Loss Plots
    plt.plot(np.arange(0, d_loss_index), d_loss, label='Discriminator')
    plt.legend()
    plt.title('GAN: Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(source_path + out_file + "_GAN_Discriminator_Loss.png")
    plt.show()
    plt.plot(np.arange(0, g_loss_index), g_loss, label='Generator')
    plt.legend()
    plt.title('GAN: Generator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(source_path + out_file + "_GAN_Generator_Loss.png")
    plt.show()

    test_loss = 0

    for data in test_loader:
        gray, real = data
        g_fake_data = G(gray)
        g_error = criterion2(g_fake_data, real)
        normalization = loss_normalizer(gray[:, :3, :, :], real)
        test_loss += g_error / normalization
    test_loss = test_loss / test_num
    print('Normalized Test Loss:', test_loss)

    G.eval()
    g_fake_data = G(test_loader.dataset.tensors[0][:4, :, :, :]).detach()

    print("Final GAN")
    display_imgs((
                 test_loader.dataset.tensors[1][:4, :, :, :].cpu(), test_loader.dataset.tensors[0][:4, 0:3, :, :].cpu(),
                 g_fake_data.cpu()), ("real", "input", "fake"), save=True,
                 fileName=source_path + out_file + '_test_imgs_model.png')

    G = torch.load(source_path + out_file + "_interm_G.pt")
    G.eval()
    print("Interm Best GAN")
    g_fake_data = G(test_loader.dataset.tensors[0][:4, :, :, :]).detach()
    display_imgs((
                 test_loader.dataset.tensors[1][:4, :, :, :].cpu(), test_loader.dataset.tensors[0][:4, 0:3, :, :].cpu(),
                 g_fake_data.cpu().detach()), ("real", "input", "fake"), save=True,
                 fileName=source_path + out_file + '_test_imgs_best_model.png')

# GAN Training