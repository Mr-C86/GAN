import torch

# a = torch.ones(2,1,1,1)
# print(a)
# b = torch.zeros(2,1,1,1)
# print(b)

# a = torch.arange(50)
# print(a)
# labens_onehot = torch.zeros((50, 10))
# print(labens_onehot.size())
# labens_onehot[torch.arange(50),5]=1
# print(labens_onehot)

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(-x) + np.exp(-x))



x = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 1#零初始化
    # w = np.random.randn(node_num, node_num) * 0.01#随机初始化
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)#Xavier初始化
    # 。
    z = np.dot(x, w)
    a = sigmoid(z)  # sigmoid函数
    activations[i] = a

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
print(activations)
