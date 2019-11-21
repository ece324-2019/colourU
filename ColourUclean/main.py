import argparse
from torch.utils.data import TensorDataset

from DataProcessing import *
from model import *

from time import time


def train_GAN (G, D, train_loader, pretraining = True, num_epochs=5, out_file=None, d_learning_rate=1e-4, g_learning_rate=1e-2):
    # Settings and Hyperparameters
    g_error_scaler = 2
    g_train_scaler = 50
    g_pretrain_epoch = 40
    d_pretrain_epoch = 40
    print_interval = 1


    # Model Parameters
    d_steps = 1
    g_steps = g_train_scaler

    # load images
    criterion = nn.BCELoss()
    criterion2 = nn.MSELoss()
    d_optimizer_pretrain = optim.Adam(D.parameters(), lr=d_learning_rate)  # , momentum=sgd_momentum)
    g_optimizer_pretrain = optim.Adam(G.parameters(), lr=g_learning_rate)  # , momentum=sgd_momentum)
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)  # , momentum=sgd_momentum)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)  # , momentum=sgd_momentum)

    d_loss = np.zeros(num_epochs * d_steps)
    g_loss = np.zeros(num_epochs * g_steps)

    d_loss_index = 0
    g_loss_index = 0

    t_init = time()

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

        # Pretraining Discriminator
        print("Pretrain Discriminator")
        for epoch in range(d_pretrain_epoch):
            print('Epoch:', epoch)
            G.eval()
            D.train()
            for data in train_loader:
                gray, real = data
                d_optimizer_pretrain.zero_grad()

                # Train discriminator on real
                d_real_decision = D(real)
                d_real_error = criterion(d_real_decision.squeeze(), torch.ones([d_real_decision.shape[0]]))
                d_real_error.backward()

                # Train on the fake
                d_fake_data = G(gray).detach()
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision.squeeze(), torch.zeros([d_fake_decision.shape[0]]))
                d_fake_error.backward()

                d_optimizer_pretrain.step()

    g_fake_data = G(train_loader.dataset.tensors[0][:4, :, :, :]).detach()
    display_imgs((train_loader.dataset.tensors[1][:4, :, :, :], g_fake_data), ("real", "fake"))

    # Train GAN
    print('Train GAN')
    for epoch in range(num_epochs):
        D.train()
        G.eval()

        d_loss_batch = []
        for data in train_loader:
            gray, real = data
            d_optimizer.zero_grad()

            # Train discriminator on real images
            d_real_decision = D(real)
            d_real_error = criterion(d_real_decision.squeeze(), torch.ones([d_real_decision.shape[0]]))
            d_real_error.backward()

            # Train on the fake images
            d_fake_data = G(gray).detach()
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision.squeeze(), torch.zeros([d_fake_decision.shape[0]]))
            d_fake_error.backward()

            d_loss_batch += [(d_real_error + d_fake_error) / (d_real_decision.shape[0] + d_fake_decision.shape[0])]

            d_optimizer.step()

        d_loss[d_loss_index] = sum(d_loss_batch)/len(d_loss_batch)
        d_loss_index += 1

        D.eval()
        G.train()

        for i in range(g_steps):
            g_loss_batch = []
            for data in train_loader:
                gray, real = data

                g_optimizer.zero_grad()

                g_fake_data = G(gray)
                dg_fake_decision = D(g_fake_data)
                g_error_1 = criterion(dg_fake_decision.squeeze(), torch.ones([dg_fake_decision.shape[0]]))  # Train G to pretend it's genuine
                g_error_2 = criterion2(g_fake_data, real)
                g_error = g_error_1 + g_error_scaler * g_error_2
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

                g_loss_batch += [g_error / dg_fake_decision.shape[0]]

            g_loss[g_loss_index] = sum(g_loss_batch)/len(g_loss_batch)
            g_loss_index += 1

        if epoch % print_interval == 0:
            print("(", time() - t_init, ") Epoch", epoch, ": D (error:", d_loss[d_loss_index - 1], ") G (error:", g_loss[g_loss_index - 1], "); ")

    torch.save(D, out_file + "_D.pt")
    torch.save(G, out_file + "_G.pt")

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

    g_fake_data = G(train_loader.dataset.tensors[0][:4, :,:,:]).detach()
    display_imgs((train_loader.dataset.tensors[1][:4, :,:,:], g_fake_data), ("real", "fake"))

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

    return (loss_whole)


def run():
    pass












if __name__ == '__main__':
    # PARSER
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--mode', type=str, default='inference')
    parser.add_argument('--user', type=str, default='none')
    parser.add_argument('--img-path', type=str, default="C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/train/Fish/")
    parser.add_argument('--in-prefix', type=str, default=None)
    parser.add_argument('--out-prefix', type=str)
    parser.add_argument('--num-imgs', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()

    # OTHER HYPERPARAMETERS
    P = {'g_hidden_size1': 32, 'g_hidden_size2': 16, 'd_input_size':64, 'd_kernel_size':3, 'd_kernel_number':20, 'd_hidden_size':16,
         'd_output_size':1, 'd_conv_layers':1, 'd_fclayers':2}

    if args.user=='mark':
        img_path = '/users/marka/Desktop/School/Engsci year 3/ECE324/project/tiny-imagenet-200/train/n01443537/images/'
    elif args.user=='alice':
        img_path = 'C:/Users/Alice/Documents/School/ECE324/Project/tiny-imagenet-200/tiny-imagenet-200/train/Fish/'
    else:
        img_path = args.img_path
    images = import_folder(img_path, args.num_imgs).float()
    grayimages = process(images)
    DT = TensorDataset(grayimages, images)
    train_loader = DataLoader(DT, batch_size=args.batch_size, shuffle=True)

    # CALL RUN IF INFERENCE MODE
    if args.mode == 'inference':
        run()

    # CALL EITHER MODEL
    else:
        if args.model == "GAN":
            G = None
            D = None
            PT = True if args.in_prefix==None else False

            if args.in_prefix == None:
                G = Generator(hidden_size1=P['g_hidden_size1'], hidden_size2=P['g_hidden_size2'])
                D = Discriminator(input_size=P['d_input_size'], kernelSize=P['d_kernel_size'], kernelNum=P['d_kernel_number'],
                                  hidden_size=P['d_hidden_size'], output_size=P['d_output_size'], convlayers=P['d_conv_layers'],
                                  fclayers=P['d_fclayers'])
            else:
                try:
                    G = torch.load(args.in_prefix + '_D.pt')
                    D = torch.load(args.in_prefix + '_G.pt')
                except:
                    print("MODELS NOT FOUND")
                    exit()

            train_GAN(G, D, train_loader, pretraining=PT, out_file=args.out_prefix, num_epochs=args.epochs)
        else:
            M = None

            if args.in_prefix == None:
                M = Autoencoder()

            else:
                try:
                    M = torch.load(args.in_prefix + '_B.pt')
                except:
                    print("MODEL NOT FOUND")
                    exit()

            train_baseline(M, train_loader, num_epochs=args.epochs)