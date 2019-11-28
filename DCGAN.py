import torch
import torch.nn as nn
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torchvision
from torchvision.utils import save_image


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, 3, 2, bias=False),  # 32
            nn.LeakyReLU(True),
            nn.Conv2d(128, 256, 4, 2, bias=False),  # 15
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 512, 4, 2, bias=False),  # 6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 256, 4, 2, bias=False),  # 2
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x)
        # print('d', out.size())
        return out


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 5, 3, 2, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv1(x)
        # print('g', out.size())
        return out


class Datas(data.Dataset):
    def __init__(self, path):
        self.dataset = []
        for i in os.listdir(path):
            x = os.path.join(path, i)
            self.dataset.append(x)
        # print(self.dataset)
        # print('1', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        strs = self.dataset[item]
        img = Image.open(strs)
        x = np.array(img)
        x = self.trans(x)

        return x

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True)])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dnet = D_Net().to(device)
gnet = G_Net().to(device)
datas = Datas(r'F:\faces')
train_data = data.DataLoader(dataset=datas, batch_size=50, shuffle=True)
loss_fun = nn.MSELoss()
dopt = torch.optim.Adam(dnet.parameters(), betas=(0.5, 0.9))
gopt = torch.optim.Adam(gnet.parameters(), betas=(0.5, 0.9))

if __name__ == '__main__':
    Epoch = 100
    SAVE_PATH = r'model/'
    SAVE_PIC = r'pic/'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not os.path.exists(SAVE_PIC):
        os.mkdir(SAVE_PIC)

for epoch in range(Epoch):
    for i, x, in enumerate(train_data):
        x = x.to(device)
        real_label = torch.ones([x.size(0), 1, 1, 1]).to(device)
        fake_label = torch.zeros([x.size(0), 1, 1, 1]).to(device)
        '''train d_net'''
        real_out = dnet(x)
        real_loss = loss_fun(real_out, real_label)

        z = torch.randn((x.size(0), 128, 1, 1)).to(device)
        fake_out = gnet(z)
        fake_img = dnet(fake_out)
        fake_loss = loss_fun(fake_img, fake_label)
        d_loss = real_loss + fake_loss
        dopt.zero_grad()
        d_loss.backward()
        dopt.step()

        '''train g_net'''
        fake_out = gnet(z)
        fake = dnet(fake_out)
        g_loss = loss_fun(fake, real_label)
        gopt.zero_grad()
        g_loss.backward()
        gopt.step()

        if i % 10 == 0:
            print('epoch:{} —》g_loss：{} —》d_loss：{}'.format(epoch, g_loss, d_loss))
            fake_out = fake_out.cpu().data
            save_image(fake_out, 'pic/fake_{}.jpg'.format(i), 10, 2, True, scale_each=True)
