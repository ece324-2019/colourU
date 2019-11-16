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


def predict():
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/test/'
    # Settings and Hyperparameters
    num_images = 100
    import_name = 'baselinev8.pt'
    loss_function = nn.MSELoss()
    model = Autoencoder()
    model.load_state_dict(torch.load(import_name))
    model.eval()

    model2 = torch.load('generator_pretrained_simple.pt')
    model2.eval()

    gen_input = import_folder(image_path, num_imgs=num_images, shuffle=True).float()
    gray_input = process(gen_input)
    baseline_fake_data = model(gray_input).detach()
    gan_fake_data = model2(gray_input).detach()

    loss_baseline = loss_function(baseline_fake_data, gen_input)
    loss_gan = loss_function(gan_fake_data, gen_input)

    print('Baseline MSELoss:', loss_baseline/num_images)
    print('GAN MSELoss:', loss_gan/num_images)

    display_imgs([gen_input.detach()[0:4, :, :, :], gray_input.detach()[0:4, :, :, :],
                           baseline_fake_data.detach()[0:4, :, :, :], gan_fake_data.detach()[0:4, :, :, :]], ['Original', 'Input', 'Baseline', 'GAN'])



predict()
