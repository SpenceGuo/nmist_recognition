import matplotlib.pyplot as plt
import torch

from torchvision import datasets, transforms


def Download_MNIST():
    # 训练集
    train_data = datasets.MNIST(
        root="./data_test/",  # 训练数据保存路径
        train=True,  # True为下载训练数据集，False为下载测试数据集
        transform=transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
        download=True,  # 是否需要下载
    )
    # 显示训练集中的第一张图片
    print(train_data.train_data.size())  # [60000,28,28]
    plt.imshow(train_data.train_data[0].numpy())
    plt.show()

    # 测试集
    test_data = datasets.MNIST(root="./data_test/", train=False)
    print(test_data.test_data.size())  # [10000, 28, 28]


if __name__ == '__main__':
    print('start')
    Download_MNIST()
    print('finished')
