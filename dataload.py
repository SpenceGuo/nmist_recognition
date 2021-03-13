import os
import torch
import struct
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from model import *


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def predict():
    all_images, all_labels = load_mnist('data/MNIST/raw/')
    temp = all_images[0].reshape(28, 28)
    plt.imshow(temp)
    plt.show()

    model = CNN_Net()
    model = torch.load('trained_models/cnn_01.pt')
    torch.no_grad()
    img_


if __name__ == '__main__':
    print('finished...')
