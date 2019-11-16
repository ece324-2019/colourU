import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
from DataProcessing import *
from Model import *



def train():
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/train/Fish/'
    # Model parameters
    fresh_start = False  # True to start new model False to continue training saved model
    d_import_name = 'discriminator_no_tanh.pt'
    g_import_name = 'generator_no_tanh.pt'
    d_export_name = 'discriminator_no_tanh.pt'
    g_export_name = 'generator_no_tanh.pt'
    g_hidden_size1 = 32
    g_hidden_size2 = 16
    d_input_size = 64    # dimension of square image
    d_kernel_size = 3    # kernel size for discriminator kernels
    d_kernel_number = 20  #number of kernels
    d_hidden_size = 16    # hidden MLP Layers
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    d_conv_layers = 2  # Number of convolutional layers
    d_fclayers = 2  # Number of fully connected layers


    d_learning_rate = 1e-4
    g_learning_rate = 1e-2
    sgd_momentum = 0.9

    num_images = 50
    num_epochs = 5

    print_interval = 1
    d_steps = 1
    g_steps = 30

    dfe, dre, ge = 0, 0, 0



    if fresh_start:
        G = Generator(hidden_size1=g_hidden_size1, hidden_size2=g_hidden_size2)
        G.float()
        D = Discriminator(input_size=d_input_size, kernelSize=d_kernel_size, kernelNum=d_kernel_number,
                      hidden_size=d_hidden_size, output_size=d_output_size, convlayers=d_conv_layers,
                      fclayers=d_fclayers)
        D.float()
    else:
        try:
            G = torch.load(g_import_name)
            D = torch.load(d_import_name)
        except:
            G = Generator(hidden_size1=g_hidden_size1, hidden_size2=g_hidden_size2)
            G.float()
            D = Discriminator(input_size=d_input_size, kernelSize=d_kernel_size, kernelNum=d_kernel_number,
                              hidden_size=d_hidden_size, output_size=d_output_size, convlayers=d_conv_layers,
                              fclayers=d_fclayers)
            D.float()

    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    criterion2 = nn.MSELoss()

    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)  #, momentum=sgd_momentum)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)  #, momentum=sgd_momentum)

    d_loss = np.zeros(num_epochs*d_steps)
    g_loss = np.zeros(num_epochs*g_steps)
    g_error_scaler = 2

    d_loss_index = 0
    g_loss_index = 0

    for epoch in range(num_epochs):
        D.train()
        G.eval()
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(import_folder(image_path, num_imgs=num_images//2, shuffle=True).float())
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision.squeeze(), Variable(torch.ones([d_real_decision.shape[0],])))  # ones = true
            # d_real_error_2 = criterion2()
            d_real_error.backward()  # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = import_folder(image_path, num_imgs=num_images//2, shuffle=True)
            d_fake_data = G(Variable(process(d_gen_input))).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision.squeeze(), Variable(torch.zeros([d_fake_decision.shape[0],])))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            d_loss[d_loss_index] = (d_real_error + d_fake_error)/(d_real_decision.shape[0]+d_fake_decision.shape[0])
            d_loss_index += 1


        D.eval()
        G.train()
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(import_folder(image_path, num_imgs=num_images, shuffle=True).float())
            g_fake_data = G(Variable(process(gen_input)))
            dg_fake_decision = D(g_fake_data)
            g_error_1 = criterion(dg_fake_decision.squeeze(), Variable(torch.ones([dg_fake_decision.shape[0],])))  # Train G to pretend it's genuine
            g_error_2 = criterion2(g_fake_data, gen_input)
            g_error = g_error_1 + g_error_scaler*g_error_2
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            g_loss[g_loss_index] = g_error/dg_fake_decision.shape[0]
            g_loss_index += 1


        if epoch % print_interval == 0:
            print("Epoch", epoch, ": D (", d_loss[d_loss_index-1], "err) G (", g_loss[g_loss_index-1], " err); ")

    torch.save(D, d_export_name)
    torch.save(G, g_export_name)

    plt.plot(np.arange(0, d_loss_index), d_loss, label='Discriminator')
    plt.legend()
    plt.title('GAN Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(np.arange(0, g_loss_index), g_loss, label='Generator')
    plt.legend()
    plt.title('GAN Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    gen_input = import_folder(image_path, num_imgs=4).float()
    g_fake_data = G(process(gen_input)).detach()
    display_processed_imgs(gen_input.detach(), g_fake_data.detach(), 4)


train()