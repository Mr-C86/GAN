import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
from GAN.FaceData import Datas
from torch.optim import lr_scheduler


class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_net = nn.Sequential(
            nn.Conv2d(3, 512, 3, 2, 1, dilation=1),
            # nn.BatchNorm2d(512),trick
            nn.LeakyReLU(0.2, True),  # 48,48

            nn.Conv2d(512, 512, 3, 2, 1, dilation=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),  # 24,24

            nn.Conv2d(512, 256, 3, 2, 1, dilation=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),  # 12,12

            nn.Conv2d(256, 128, 3, 2, 1, dilation=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),  # 6,6

            nn.Conv2d(128, 64, 3, 2, 1, dilation=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),  # 3,3

            nn.Conv2d(64, 1, 3, 1, 0, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.d_net(x)
        # print('1size', out.size())
        return out


class G_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_net = nn.Sequential(

            nn.ConvTranspose2d(512, 64, 3, 2, 0, 0, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 128, 3, 2, 1, 1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 256, 3, 2, 1, 1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 512, 3, 2, 1, 1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 512, 3, 2, 1, 1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 3, 3, 2, 1, 1, dilation=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.g_net(x)
        # print('size', out.size())
        return out


if __name__ == '__main__':
    SAVE_PATH_D_NET = 'model/face_gan_deeplab_d_net.pkl'
    SAVE_PATH_G_NET = 'model/face_gan_deeplab_g_net.pkl'
    EPOCH = 30
    BATCH_SIZE = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datas = Datas('facesdata')
    train_data = data.DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.MSELoss(reduction='mean')

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    if os.path.exists(SAVE_PATH_D_NET):
        d_net.load_state_dict(torch.load(SAVE_PATH_D_NET))
        print('加载d_net--ing')
    if os.path.exists(SAVE_PATH_G_NET):
        g_net.load_state_dict(torch.load(SAVE_PATH_G_NET))
        print('加载g_net--ing')
    s = SummaryWriter()

    d_opt = torch.optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))  # betas损失的平滑程度
    # scheduler = lr_scheduler(d_opt, 20,0.1)
    g_opt = torch.optim.SGD(g_net.parameters(), lr=0.0002, )  #betas=(0.5, 0.999)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_data):
            real_label = torch.ones(x.size(0), 1, 1, 1).to(device)
            fake_label = torch.zeros(x.size(0), 1, 1, 1).to(device)
            # print(real_label.size())
            # print(fake_label.size())
            # 训练判别器
            real_out = d_net(x.to(device))
            d_real_loss = loss_fn(real_out, real_label)

            z = torch.randn(x.size(0), 512, 1, 1).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_fake_loss = loss_fn(fake_out, fake_label)
            d_loss = d_fake_loss + d_real_loss

            d_opt.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(d_net.parameters(), max_norm=20, norm_type=2)  # 对更新后的权重强制截断到一定范围内
            d_opt.step()

            # 训练判别器
            c = torch.randn(x.size(0), 512, 1, 1).to(device)
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
                save_image(x, 'pic/deeplab/real_face_deeplab_{}.jpg'.format(i), nrow=10, normalize=True,
                           scale_each=True)
                save_image(fake_imgs, 'pic/deeplab/fake_face_deeplab_{}.jpg'.format(i), nrow=10, normalize=True,
                           scale_each=True)

        s.add_histogram('face_deeplab_d_w', d_net.d_net[0].weight.data, global_step=epoch)
        s.add_histogram('face_deeplab_g_w', g_net.g_net[0].weight.data, global_step=epoch)
        s.add_scalars('face_deeplab_loss', {'face_deeplab_d_loss': d_loss, 'face_deeplab_g_loss': g_loss},
                      global_step=epoch)
        torch.save(d_net.state_dict(), SAVE_PATH_D_NET)
        torch.save(g_net.state_dict(), SAVE_PATH_G_NET)
