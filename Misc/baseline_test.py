import torch
from torch.utils.data import TensorDataset

from DataProcessing import *
from model import *

def train(model, data, num_epochs=5,  batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # <--
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

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

        loss_whole += [sum(loss_batch)/len(loss_batch)]

        print('({:.1f}) Epoch:{}, Loss:{:.4f}'.format(time() - t_init, epoch+1, float(loss)))

    return(loss_whole)

np.random.seed(0)
torch.manual_seed(0)

T = False

path = "/users/marka/Desktop/School/Engsci year 3/ECE324/project/tiny-imagenet-200/val/images/"
img_tensor = import_folder(path, 1000)
gray_img_tensor = process(img_tensor)

print(img_tensor.size())

if T:
    DT = TensorDataset(gray_img_tensor, img_tensor)
    model = Autoencoder()
    loss = train(model, DT, num_epochs=50)

    plt.plot(loss)
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.title("Loss curve for Baseline model")
    plt.show()

    torch.save(model,"baselinev10.pt")
else:
    model = torch.load("baselinev10.pt")
    transformed = model(gray_img_tensor[40:45, :, :, :]).detach()
    display_imgs((img_tensor[40:45, :, :, :], gray_img_tensor[100:10    5, :, :, :]), ("Original", "Processed"))
