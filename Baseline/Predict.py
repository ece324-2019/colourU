import torch
from torch.utils.data import TensorDataset, DataLoader

from DataProcessing import *
from model import *


def run():
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/train/Fish/'
    # Settings and Hyperparameters
    fresh_start = False  # True to start new model False to continue training saved model
    import_name = 'baseline.pt'
    num_images = 30


    model = torch.load(import_name)
    print(import_name, 'loaded')
    img_tensor = import_folder(image_path, num_imgs=num_images).float()
    gray_img_tensor = process(img_tensor)
    transformed_imgs = model(gray_img_tensor[11:15, :, :, :])

    display_processed_imgs(img_tensor[11:15, :, :, :], transformed_imgs.detach(), 4)

run()