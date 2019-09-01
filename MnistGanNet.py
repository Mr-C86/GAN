import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os


class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.d_net(x)
        return out


class G_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 784),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.g_net(x)
        return out


if __name__ == '__main__':
    SAVE_PATH_D_NET = 'model/mnist_gan_d_net.pkl'
    SAVE_PATH_G_NET = 'model/mnist_gan_g_net.pkl'
    EPOCH = 20
    BATCH_SIZE = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datas = torchvision.datasets.MNIST(root='datasets', train=True,
                                       transform=torchvision.transforms.ToTensor(), download=True)
    train_data = data.DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.BCELoss()

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    if os.path.exists(SAVE_PATH_D_NET):
        d_net.load_state_dict(torch.load(SAVE_PATH_D_NET))
        print('加载d_net--ing')
    if os.path.exists(SAVE_PATH_G_NET):
        g_net.load_state_dict(torch.load(SAVE_PATH_G_NET))
        print('加载g_net--ing')
    s = SummaryWriter()

    d_opt = torch.optim.Adam(d_net.parameters())
    g_opt = torch.optim.Adam(g_net.parameters())

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_data):
            img = x.reshape(x.size(0), -1)
            real_label = torch.ones(x.size(0), 1).to(device)
            fake_label = torch.zeros(x.size(0), 1).to(device)
            # 训练判别器
            real_out = d_net(img.to(device))
            d_real_loss = loss_fn(real_out, real_label)

            z = torch.randn(x.size(0), 128).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_fake_loss = loss_fn(fake_out, fake_label)
            d_loss = d_fake_loss + d_real_loss

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # 训练判别器
            c = torch.randn(x.size(0), 128).to(device)
            fake_imgs = g_net(c)
            output = d_net(fake_imgs)
            g_loss = loss_fn(output, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                print('epoch: {} | i/len: {}/{} | d_loss: {:.3f} | g_loss: {:.3f} | d_score: {:.3f} | g_score: {:.3f}'
                      .format(epoch, i, len(train_data), d_loss, g_loss, real_out.data.mean(), fake_out.data.mean()))

        real_img = img.reshape([-1, 1, 28, 28])
        fake_imgs = fake_imgs.cpu().reshape([-1, 1, 28, 28])
        save_image(real_img, 'pic/mnist/real_{}.jpg'.format(epoch), nrow=10, normalize=True, scale_each=True)
        save_image(fake_imgs, 'pic/mnist/fake_{}.jpg'.format(epoch), nrow=10, normalize=True, scale_each=True)

        s.add_histogram('d_w', d_net.d_net[0].weight.data, global_step=epoch)
        s.add_histogram('g_w', g_net.g_net[0].weight.data, global_step=epoch)
        s.add_scalars('loss', {'d_loss': d_loss, 'g_loss': g_loss}, global_step=epoch)
        torch.save(d_net.state_dict(), SAVE_PATH_D_NET)
        torch.save(g_net.state_dict(), SAVE_PATH_G_NET)
