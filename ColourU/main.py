import argparse
from torch.utils.data import TensorDataset

from DataProcessing import *
from model import *

from time import time
import numpy as np

from training_functions import *


def train_baseline (model, train_loader, num_epochs=5, learning_rate=1e-3):
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    t_init = time()
    loss_whole = []
    for epoch in range(num_epochs):
        loss_batch = []
        for data in train_loader:
            img_in, img_out = data
            recon = model(img_in)
            loss = criterion(recon, img_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_batch += [loss]

        loss_whole += [sum(loss_batch) / len(loss_batch)]

        print('({:.1f}) Epoch:{}, Loss:{:.4f}'.format(time() - t_init, epoch + 1, float(loss)))

    return loss_whole


def run(model, data_loader, save):
    model.eval()
    for data in data_loader:
        gray, real = data
        generated = G(gray)
    print(generated.shape)
    print(data_loader.dataset.tensors[0].shape)
    print(data_loader.dataset.tensors[1].shape)
    display_imgs((data_loader.dataset.tensors[1][:,:,:,:].cpu(), data_loader.dataset.tensors[0][:,0:3,:,:].cpu(), generated.cpu().detach()), ("real", "input", "fake"), save=True, fileName=save)

def evaluate(model_path_g, model_path_b, data_path, data_num, savelocation_g, savelocation_b, size='Small'):
    if size=='small' or size=='Small':
        images = import_folder(data_path, data_num, expected_size=(64, 64, 3)).float()
    else:
        images = import_folder(data_path, data_num, expected_size=(256, 256, 3)).float()
    grayimages = process(images)
    DT = TensorDataset(grayimages, images)
    data_loader = DataLoader(DT, batch_size=1, shuffle=False)
    G=torch.load(model_path_g)
    B=torch.load(model_path_b)
    gen_imgs_g=[]
    loss_array_g=[]
    gen_imgs_b=[]
    loss_array_b=[]
    loss_function = nn.MSELoss()
    for data in data_loader:
        gray, real = data
        generated = G(gray)
        gen_imgs_g.append(generated.cpu().detach())
        loss_array_g.append((loss_function(generated, real)/loss_function(gray[:,:3,:,:], real)).item())
        generated = B(gray)
        gen_imgs_b.append(generated.cpu().detach())
        loss_array_b.append((loss_function(generated, real)/loss_function(gray[:,:3,:,:], real)).item())

    generated_images=np.stack(gen_imgs_g, axis=2)
    generated_images = torch.tensor(np.transpose(generated_images.squeeze(), [1,0,2,3])).float()
    display_imgs((data_loader.dataset.tensors[1][:,:,:,:].cpu(), data_loader.dataset.tensors[0][:,0:3,:,:].cpu(), generated_images.cpu().detach()), ("real", "input", "fake"), save=True, fileName=savelocation_g)
    print(loss_array_g)
    generated_images=np.stack(gen_imgs_b, axis=2)
    generated_images = torch.tensor(np.transpose(generated_images.squeeze(), [1,0,2,3])).float()
    display_imgs((data_loader.dataset.tensors[1][:,:,:,:].cpu(), data_loader.dataset.tensors[0][:,0:3,:,:].cpu(), generated_images.cpu().detach()), ("real", "input", "fake"), save=True, fileName=savelocation_b)
    print(loss_array_b)


if __name__ == '__main__':
    # PARSER
    model = 'GAN'
    mode = 'evaluate'  # 'train', 'evaluate' or 'inference'
    source_path = '/content/drive/My Drive/ECE324 Project/'  # root folder

    # Training only parameters
    img_path = '10000/train/'  # path from source to training images, must end in '/'
    val_path = '10000/val/'  # path from source to validation images, must end in '/'
    test_path = '10000/val_test/'  # path from source to test images, must end in '/'
    in_prefix = None  # Optional: prefix for importing files saved with this naming convention
    out_prefix = '10000'  # prefix for saving files
    num_imgs = 8000  # number of training images
    val_num_imgs = 1600  # number of validation images
    test_num_imgs = 400  # number of test images
    batch_size = 1000
    epochs = 20

    # Evaluate Only Parameters
    g_model = '/content/drive/My Drive/ECE324 Project/10000_interm_G.pt'  # path to the GAN generator
    b_model = '/content/drive/My Drive/ECE324 Project/five_categories_baseline.pt'  # path to the baseline model
    img_source = '/content/drive/My Drive/ECE324 Project/Eval Images/salamander/'  # source of images
    GAN_results = '/content/drive/My Drive/ECE324 Project/Eval Images/GAN_results/GAN_new_small.png'  # save location of GAN results
    baseline_results = '/content/drive/My Drive/ECE324 Project/Eval Images/baseline_results/baseline_new_small.png'  # save location of baseline results
    eval_img_size = 'small'  # 'small' for 64x64 pixels, 'large' for 256x256 pixels

    # Inference Only Parameters
    inf_img = '/content/drive/My Drive/ECE324 Project/Test_images/Small/'  # source for inference images
    inf_num_imgs = 3  # number of images to infer note img size is 64x64 pixels
    model_source = '/content/drive/My Drive/ECE324 Project/five_categories_G.pt'
    inf_output = '/content/drive/My Drive/ECE324 Project/Test_images/Large/final_model_new_small.png'

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    np.random.seed(4)
    torch.manual_seed(4)

    # OTHER HYPERPARAMETERS
    P = {'g_hidden_size1': 32, 'g_hidden_size2': 64, 'g_hidden_size3': 128, 'g_hidden_size4': 256, 'd_input_size': 64,
         'd_kernel_size': 5, 'd_kernel_number': 64, 'd_hidden_size': 64,
         'd_output_size': 1, 'd_conv_layers': 3, 'd_fclayers': 2}

    if mode != 'inference' and mode != 'evaluate':  # loads data
        images = import_folder(source_path + img_path, num_imgs).float()
        grayimages = process(images.cpu())
        if torch.cuda.is_available():
            grayimages.cuda()
            images.cuda()
        DT = TensorDataset(grayimages, images)
        train_loader = DataLoader(DT, batch_size=batch_size, shuffle=True)
        train_size = len(images)
        images = import_folder(source_path + val_path, val_num_imgs).float()
        grayimages = process(images.cpu())
        if torch.cuda.is_available():
            grayimages.cuda()
            images.cuda()
        DT = TensorDataset(grayimages, images)
        val_loader = DataLoader(DT, batch_size=batch_size, shuffle=True)
        images = import_folder(source_path + test_path, test_num_imgs).float()
        grayimages = process(images.cpu())
        if torch.cuda.is_available():
            grayimages.cuda()
            images.cuda()
        DT = TensorDataset(grayimages, images)
        test_loader = DataLoader(DT, batch_size=batch_size, shuffle=True)

    # call run if inference mode
    if mode == 'evaluate':
        evaluate(g_model, b_model, img_source, 4, GAN_results, baseline_results, size=eval_img_size)

    elif mode == 'inference':
        images = import_folder(inf_img, inf_num_img, expected_size=(64, 64, 3)).float()
        grayimages = process(images)
        DT = TensorDataset(grayimages, images)
        data = DataLoader(DT, batch_size=3, shuffle=False)
        model = torch.load(model_source)
        run(model, data, inf_output)

    # call either model
    else:
        if model == "GAN":
            G = None
            D = None
            PT = True if in_prefix == None else False

            if in_prefix == None:
                G = GenResNet(hidden_size1=P['g_hidden_size1'], hidden_size2=P['g_hidden_size2'],
                              hidden_size3=P['g_hidden_size3'], hidden_size4=P['g_hidden_size4'])
                D = Discriminator(input_size=P['d_input_size'], kernelSize=P['d_kernel_size'],
                                  kernelNum=P['d_kernel_number'],
                                  hidden_size=P['d_hidden_size'], output_size=P['d_output_size'],
                                  convlayers=P['d_conv_layers'],
                                  fclayers=P['d_fclayers'])
            else:
                try:
                    G = torch.load(source_path + in_prefix + '_G.pt')
                    D = torch.load(source_path + in_prefix + '_D.pt')
                except:
                    print("MODELS NOT FOUND")
                    exit()

            if torch.cuda.is_available():
                G.cuda()
                D.cuda()

            train_GAN(G, D, train_loader, val_loader, test_loader, num_imgs, val_num_imgs, test_num_imgs, source_path,
                      pretraining=PT, out_file=out_prefix, num_epochs=epochs, d_learning_rate=0.004,
                      g_learning_rate=0.001, m_param=P)
        elif model == 'baseline':
            M = None

            if in_prefix == None:
                M = Autoencoder()

            else:
                try:
                    M = torch.load(source_path + in_prefix + '_B.pt')
                except:
                    print("MODEL NOT FOUND")
                    exit()

            if torch.cuda.is_available():
                M.cuda()

            train_baseline(M, train_loader, val_loader, test_loader, num_imgs, val_num_imgs, test_num_imgs, source_path,
                           out_prefix, num_epochs=epochs)