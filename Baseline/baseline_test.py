import torch
from torch.utils.data import TensorDataset, DataLoader

from DataProcessing import *
from model import *
import argparse


def run(image_path, import_name, export_name, num_images, num_epochs, fresh_start=True, save=True):

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
    if save:
        torch.save(model, export_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='null')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_images', type=int, default=100)

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    if args.path == 'null':
        image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/val/'
    else:
        image_path = args.path

    # Settings and Hyperparameters
    fresh_start = False  # True to start new model False to continue training saved model
    import_name = 'baseline_fish.pt'
    export_name = 'baseline_fish.pt'
    num_images = args.num_images
    num_epochs = args.epochs

    run(image_path, import_name, export_name, num_images, num_epochs, fresh_start=fresh_start, save=True)
