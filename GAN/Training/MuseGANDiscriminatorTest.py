import torch

from GAN.Models.MuseGANDiscriminator import MuseGANDiscriminator
from Data import lpd_loader


def random_data():
    return torch.randn((32, 1, 384, 84))


if __name__ == '__main__':
    epochs = 10

    discriminator = MuseGANDiscriminator()
    data = torch.tensor(lpd_loader.sample_lpd5(500))

    labels = torch.cat((torch.ones(32), torch.zeros(32)))[:, None]

    optim = torch.optim.Adam(discriminator.parameters())
    criterion = torch.nn.BCELoss()

    for i in range(epochs):
        loss_sum = 0
        for batch in range(0, len(data)-32, 32):
            real = data[batch:batch+32][:, None, :, :]
            rand = random_data()

            inputs = torch.cat((real, rand))

            optim.zero_grad()
            result = discriminator(inputs)

            loss = criterion(result, labels)
            loss_sum += loss.item()
            print(f"Epoch {i+1}, Batch {(batch//32)+1}")
            loss.backward()
            optim.step()
        print(f"Epoch {i+1}: Loss = {loss_sum}")
