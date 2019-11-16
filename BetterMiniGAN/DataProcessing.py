import torch
import torch.utils.data as data
import numpy as np

from skimage.color import rgb2lab, lab2rgb
from skimage import io
from matplotlib import pyplot as plt

from os import listdir, remove

# Imports a certain number of images from a folder
def import_folder(path, num_imgs=-1, start = 0, shuffle=True):
    if shuffle:
        img_files = np.asarray(listdir(path))
        np.random.shuffle(img_files)
    else:
        img_files = np.asarray(listdir(path))
    if num_imgs == -1:
        num_imgs = len(img_files)
    img_array_stack = []
    array = np.zeros((64, 64, num_imgs, 3))
    for i in range(start, start + num_imgs):
        if len(io.imread(path + img_files[i]).shape) != 3:
            remove(path + img_files[i])
            print(img_files[i], " removed")
        else:
            img_array_stack += [io.imread(path + img_files[i])]

    stack = np.stack(img_array_stack, axis=2)
    convert = rgb2lab(stack)
    return torch.tensor(convert).permute(2, 3, 0, 1)

# Processes the images
def process (LABimgs, all_gray=False):
    H = LABimgs.size()[2]
    W = LABimgs.size()[3]

    GrayImgs = LABimgs.clone()
    GrayImgs[:, 1:, :, :] = 0

    if all_gray:
        return GrayImgs.float()

    num_pixels = np.random.geometric(0.125, LABimgs.size()[0])

    for img in range (LABimgs.size()[0]):
        size = np.random.randint(1, 10, size=num_pixels[img])
        pixels = np.random.normal(loc = [H/2, W/2], scale = [H/4, W/4], size=[num_pixels[img],2]).astype(int).clip(min=0, max=min(H,W) - max(size))

        for px in range(num_pixels[img]):
            avg_col = torch.mean(LABimgs[img, 1:, pixels[px,0]:pixels[px,0] + size[px], pixels[px,1]:pixels[px,1] + size[px]], dim=[1,2])
            GrayImgs[img, 1:, pixels[px,0]:pixels[px,0] + size[px], pixels[px,1]:pixels[px,1] + size[px]] = avg_col.repeat(size[px], size[px], 1).permute(2, 0, 1)

    return GrayImgs.float()

# Displays processed images, for verificaiton
def display_processed_imgs (img_tensor, gray_img_tensor, num_imgs):
    fig = plt.figure()

    for i in range(num_imgs):
        img = img_tensor[i, :, :, :].permute(1, 2, 0)
        fig.add_subplot(2, num_imgs, i+1)
        plt.imshow(lab2rgb(img, illuminant = "D50"))
        img = gray_img_tensor[i, :, :, :].permute(1,2,0)
        fig.add_subplot(2, num_imgs, num_imgs + i+1)
        plt.imshow(lab2rgb(img, illuminant="D50"))

    plt.show()
