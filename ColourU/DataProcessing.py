import torch
import numpy as np

from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot as plt

from os import listdir

# Imports a certain number of images from a folder
def import_folder(path, num_imgs=-1, start = 0, expected_size=(64, 64, 3)):
    img_files = listdir(path)

    if num_imgs == -1:
        num_imgs = len(img_files)

    img_array_stack = []
    for i in range(start, start + num_imgs):
        img = plt.imread(path + img_files[i])
        if img.shape == expected_size:
            img_array_stack += [plt.imread(path + img_files[i])]

    return torch.tensor(rgb2lab(np.stack(img_array_stack, axis=2))).permute(2, 3, 0, 1).float()

# Processes the images
def process (LABimgs, all_gray=False):
    H = LABimgs.size()[2]
    W = LABimgs.size()[3]
    n = LABimgs.size()[0]

    GrayImgs = torch.zeros((n, 4, H, W)).cpu()
    GrayImgs[:, 0, :, :] = LABimgs[:, 0, :, :]

    if all_gray:
        return GrayImgs.float()

    num_pixels = np.random.geometric(0.125, LABimgs.size()[0])
    full_colour = torch.zeros(n)
    full_colour[0:int(n/100)] = 1
    full_colour = full_colour[torch.randperm(n)]

    for img in range (LABimgs.size()[0]):
        if (full_colour[img] == 0):
            size = np.random.randint(1, 10, size=num_pixels[img])
            pixels = np.random.normal(loc = [H/2, W/2], scale = [H/4, W/4], size=[num_pixels[img],2]).astype(int).clip(min=0, max=min(H,W) - max(size))

            for px in range(num_pixels[img]):
                avg_col = torch.mean(LABimgs[img, 1:, pixels[px,0]:pixels[px,0] + size[px], pixels[px,1]:pixels[px,1] + size[px]], dim=[1,2])
                GrayImgs[img, 1:3, pixels[px,0]:pixels[px,0] + size[px], pixels[px,1]:pixels[px,1] + size[px]] = avg_col.repeat(size[px], size[px], 1).permute(2, 0, 1)
        else:
            GrayImgs[img, 1:3, :, :] = LABimgs[img, 1:3, :, :]

        highpass = GrayImgs[img, 0, :, :] - torch.tensor(ndimage.gaussian_filter(GrayImgs[img,0,:,:], 3)).cpu()
        GrayImgs[img,3,:,:] = highpass
    if torch.cuda.is_available():
      return GrayImgs.cuda().float()
    else:
      return GrayImgs.float()

# Displays Images
def display_imgs (img_tensor_list, labels, save=False, fileName='image.png'):
    fig = plt.figure()

    num_sets = len(img_tensor_list)
    num_imgs = img_tensor_list[0].size()[0]

    for set in range(num_sets):
        for img in range(num_imgs):
            image = img_tensor_list[set][img, :, :, :].permute(1,2,0)
            figure = fig.add_subplot(num_sets, num_imgs, num_imgs*set + img + 1)

            if img == 0:
                figure.set_ylabel(labels[set])

            plt.imshow(lab2rgb(image))
    if save:
      plt.savefig(fileName)
    plt.show()

# Data Processing