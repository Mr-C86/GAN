import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
from GAN.FaceData import Datas
from GAN.ranger.ranger import Ranger


class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, 0, bias=False),
            # nn.BatchNorm2d(512),trick
            nn.PReLU(),
            # nn.MaxPool2d(2),#170

            nn.Conv2d(64, 128, 5, 3, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.MaxPool2d(2),  # 56

            nn.Conv2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),  # 28

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),  # 14

            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),  # 7

            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),  # 3.5

            nn.Conv2d(128, 1, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.d_net(x)
        # print('1size', out.size())
        return out


class G_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_net = nn.Sequential(

            nn.ConvTranspose2d(128, 128, 3, 1, 0, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),  # 3.5

            nn.ConvTranspose2d(128, 128, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 128, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.ConvTranspose2d(128, 64, 5, 3, 0, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 3, 5, 3, 0, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.g_net(x)
        # print('size', out.size())
        return out


if __name__ == '__main__':
    SAVE_PATH_D_NET = 'model/face_gan_d_net_ranger_512_HD.pkl'
    SAVE_PATH_G_NET = 'model/face_gan_g_net_ranger_512_HD.pkl'
    EPOCH = 100
    BATCH_SIZE = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datas = Datas('HDdata')
    train_data = data.DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    loss_fn = nn.BCELoss(reduction='sum')  #

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    if os.path.exists(SAVE_PATH_D_NET):
        d_net.load_state_dict(torch.load(SAVE_PATH_D_NET))
        print('加载d_net--ing')
    if os.path.exists(SAVE_PATH_G_NET):
        g_net.load_state_dict(torch.load(SAVE_PATH_G_NET))
        print('加载g_net--ing')
    s = SummaryWriter()

    d_opt = torch.optim.SGD(d_net.parameters(), lr=0.001,)  # betas损失的平滑程度 betas=(0.5, 0.999)
    g_opt = torch.optim.SGD(g_net.parameters(), lr=0.001, )  #betas=(0.5, 0.999)

    # model, optimizer = amp.initialize(d_net, d_opt, opt_level="O1")  # 这里是“欧一”，不是“零一”amp加速
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_data):
            real_label = torch.ones(x.size(0), 1, 1, 1).to(device)
            fake_label = torch.zeros(x.size(0), 1, 1, 1).to(device)
            # print(real_label.size())
            # print(fake_label.size())
            # 训练判别器
            real_out = d_net(x.to(device))
            # print(real_label.size())
            # print(real_out.size())
            d_real_loss = loss_fn(real_out, real_label)

            z = torch.randn(x.size(0), 128, 1, 1).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_fake_loss = loss_fn(fake_out, fake_label)
            d_loss = d_fake_loss + d_real_loss

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # 训练判别器
            c = torch.randn(x.size(0), 128, 1, 1).to(device)
            fake_imgs = g_net(c)
            output = d_net(fake_imgs)
            g_loss = loss_fn(output, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                print('epoch: {} | i/len: {}/{} | d_loss: {:.3f} | g_loss: {:.3f} | d_score: {:.3f} | g_score: {:.3f}'
                      .format(epoch, i, len(train_data), d_loss, g_loss, real_out.data.mean(), fake_out.data.mean()))

                # real_img = x.reshape([-1, 3, 96, 96])
                # fake_imgs = fake_imgs.cpu().reshape([-1, 3, 96, 96])
                # 将图片*0.5+0.5*255
                save_image(x, 'pic/512/real_face_ranger_{}.jpg'.format(i), nrow=10, normalize=True, scale_each=True)
                save_image(fake_imgs, 'pic/512/fake_face_ranger_{}.jpg'.format(i), nrow=10, normalize=True,
                           scale_each=True)

        # s.add_histogram('face_d_w_ranger', d_net.d_net[0].weight.data, global_step=epoch)
        # s.add_histogram('face_g_w_ranger', g_net.g_net[0].weight.data, global_step=epoch)
        # s.add_scalars('face_loss_ranger', {'face_d_loss': d_loss, 'face_g_loss_ranger': g_loss}, global_step=epoch)
        torch.save(d_net.state_dict(), SAVE_PATH_D_NET)
        torch.save(g_net.state_dict(), SAVE_PATH_G_NET)
