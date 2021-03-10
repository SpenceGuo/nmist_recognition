import numpy as np
from dataload import *


if __name__ == '__main__':
    images, labels = load_mnist('data/MNIST/raw/')
    print(images[0])
    print(len(images))
    print(len(labels))
