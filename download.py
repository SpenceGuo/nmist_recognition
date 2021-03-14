from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch


def Download_MNIST():
    train_data = datasets.MNIST(
        root='./data_test/', # 训练数据保存路径
        train=True, # True为下载训练数据集，False为下载测试数据集
        transform=transforms.ToTensor(), # 数据范围已从(0-255)压缩到(0,1)
        download=True # 是否需要下载
    )

    # 显示训练集中的第一张图片
    print(train_data.train_data.size())  # [60000,28,28]
    plt.imshow(train_data.train_data[0].numpy())
    plt.show()

    # test_data = datasets.MNIST(root="./data/", train=False)
    #
    # test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255
    # test_y = test_data.test_labels


if __name__ == '__main__':
    print('Downloading the dataset, please wait...')
    Download_MNIST()
    print('Finished!')
