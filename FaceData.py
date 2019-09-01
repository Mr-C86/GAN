import torch
import torch.nn as nn
import os
from PIL import Image
import torch.utils.data as data
import numpy as np


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
        img = torch.Tensor((((np.array(img).transpose(2, 0, 1)) / 255) - 0.5) / 0.5)
        return img, label

# d = Datas('facesdata')
# d[200]
