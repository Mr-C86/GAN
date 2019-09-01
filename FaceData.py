import torch
import torch.nn as nn
import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision

transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
class Datas(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        strs = self.dataset[item]
        label = strs.split('.')[0][-1]
        img = Image.open(os.path.join(self.path, strs))
        img = transform(np.array(img))#归一化到-1-1之间                           
        img = torch.Tensor(img)
        return img, label

# d = Datas('facesdata')
# d[200]
