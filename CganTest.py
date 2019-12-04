import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from GAN.MnistCGanNet import G_Net, D_Net
from PIL import Image

SAVE_PATH_D_NET = 'model/face_gan_d_net_cgan.pkl'
SAVE_PATH_G_NET = 'model/face_gan_g_net_cgan.pkl'
num_img = 10
g = G_Net()
d = D_Net()

g.load_state_dict(torch.load(SAVE_PATH_G_NET))
d.load_state_dict(torch.load(SAVE_PATH_D_NET))

img_list = []
for i in range(10):
    z = torch.randn((num_img, 118))
    x = torch.zeros((num_img, 10))
    x[:, i] = 1
    z = torch.cat((z, x), dim=1)
    z = z.reshape(z.size(0), -1, 1, 1)

    fake_img = g(z)
    img_arr = (fake_img.data.numpy() * 0.5 + 0.5) * 255
    img = Image.fromarray(img_arr.reshape(280, 28))
    plt.imshow(img)
    plt.pause(0.1)
    img_list.append(fake_img.data.tolist())
    img_tensor = torch.tensor(img_list)

images = img_tensor.reshape([100, 1, 28, 28])
save_image(images, 'pic/cgantest/cgan_fake_img_{}.png'.format('img'), nrow=10, normalize=True, scale_each=True)
