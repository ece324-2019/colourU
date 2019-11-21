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
        num_kernels = [16, 32, 64, 128]
        kernel_size = [3, 3, 7,3]
        model = Autoencoder(num_kernels, kernel_size)
    else:
        try:
            model = torch.load(import_name)
            print(import_name, 'loaded')
        except:
            num_kernels = [16, 32, 64, 128]
            kernel_size = [3, 3, 7, 3]
            model = Autoencoder(num_kernels, kernel_size)

    transformed_imgs = model(gray_img_tensor[11:15, :, :, :])

    display_processed_imgs(img_tensor[11:15, :, :, :], transformed_imgs.detach(), 4)

    model, min_val_loss = train(model, DT, num_epochs=num_epochs)
    print("Min Val Loss:", min_val_loss)

    transformed_imgs = model(gray_img_tensor[11:15, :, :, :])

    display_processed_imgs(img_tensor[11:15, :, :, :], transformed_imgs.detach(), 4)
    if save:
        torch.save(model, export_name)


def hyperparameter_search(num_images, val_num_images):
    # initialize data
    image_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/val/'
    val_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/test/'
    img_tensor = import_folder(image_path, num_imgs=num_images).float()
    gray_img_tensor = process(img_tensor)
    train_data = TensorDataset(gray_img_tensor, img_tensor)
    img_tensor = import_folder(val_path, num_imgs=val_num_images).float()
    gray_img_tensor = process(img_tensor)
    val_data = TensorDataset(gray_img_tensor, img_tensor)

    #initialize model parameters
    kernel_nums = [[16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128], [16, 32, 64, 128],
                   [128, 64, 32, 16], [16, 32, 32, 16], [128, 64, 64, 128]]
    kernel_sizes = [[3, 3, 3, 3], [5, 5, 5, 5], [7, 7, 7, 7], [3, 3, 5, 7], [7, 5, 3, 3], [3, 5, 5, 3], [7, 5, 5, 7]]

    kernel_num_losses = np.zeros(len(kernel_nums))
    kernel_size_losses = np.zeros(len(kernel_sizes))

    for i in range(len(kernel_nums)):
        model = Autoencoder(kernel_nums[i], kernel_sizes[0])
        model, loss = train(model, train_data, val_data, num_images, val_num_images, num_epochs=15,
                            batch_size=64, name=('Kernel_Num_Set_'+str(i)))
        kernel_num_losses[i] = loss
    print(kernel_num_losses)
    plt.plot(np.arange(len(kernel_nums)), kernel_num_losses)
    plt.title('Hyperparameter Search: Number of Kernels')
    plt.xlabel('Kernel Numbers')
    plt.ylabel('Loss')
    plt.show()

    for i in range(len(kernel_sizes)):
        model = Autoencoder(kernel_nums[0], kernel_sizes[i])
        model, loss = train(model, train_data, val_data, num_images, val_num_images, num_epochs=15, batch_size=64, name=('Kernel_Size_Set_'+str(i)))
        kernel_num_losses[i] = loss

    plt.plot(np.arange(len(kernel_sizes)), kernel_size_losses)
    plt.title('Hyperparameter Search: Kernel Sizes')
    plt.xlabel('Kernel Size')
    plt.ylabel('Loss')
    plt.show()


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
    hyperparameter_search(100, 50)
    #run(image_path, import_name, export_name, num_images, num_epochs, fresh_start=fresh_start, save=True)
