import torch
from torch.utils.data import TensorDataset, DataLoader

from DataProcessing import *
from model import *

def run():
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/train/Fish/'
    # Settings and Hyperparameters
    fresh_start = False  # True to start new model False to continue training saved model
    import_name = 'baseline_fish.pt'
    export_name = 'baseline_fish.pt'
    num_images = 100
    num_epochs = 50

    img_tensor = import_folder(image_path, num_imgs=num_images).float()
    gray_img_tensor = process(img_tensor)
    DT = TensorDataset(gray_img_tensor, img_tensor)
    if fresh_start:
        model = Autoencoder()
    else:
        try:
            model = torch.load(import_name)
            print(import_name, 'loaded')
        except:
            model = Autoencoder()

    transformed_imgs = model(gray_img_tensor[11:15, :, :, :])

    display_processed_imgs(img_tensor[11:15, :, :, :], transformed_imgs.detach(), 4)

    model = train(model, DT, num_epochs=num_epochs)

    transformed_imgs = model(gray_img_tensor[11:15, :, :, :])

    display_processed_imgs(img_tensor[11:15, :, :, :], transformed_imgs.detach(), 4)

    torch.save(model, export_name)

run()