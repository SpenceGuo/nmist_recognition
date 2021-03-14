import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 输入的维度：in_dim；
# 第一层神经网络的神经元个数n_hidden_1；
# 第二层神经网络神经元的个数n_hidden_2,out_dim
# 第三层网络(输出成)神经元的个数
class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out


# nn.Sequential将网络的层组合到一起里面,按顺序进行网络构建。
# 激活层和池化从都不需要参数。
class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out


class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out


class CNN_Net(nn.Module):
    def __init__(self):
        # 继承__init__() 功能
        super(CNN_Net, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            # in_channels:输入通道数  out_channels:输出通道数
            # kernel_size:卷积核尺寸  stride:步长 padding:填充
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # 经过卷积层 输出[16,28,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 经过池化 输出[16,14,14] 传入下一个卷积
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 经过池化输出
            # 传入输出层
        )
        # 输出层：in_feature为上一层卷积层输出大小  out_feature为输出的分类类别数
        self.output = nn.Linear(in_features=32*7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)  # 输出[50,10]
        return output


def Linear_Net_models():
    # 定义超参数
    batch_size = 32
    learning_rate = 0.02

    # 定义图片格式转化为Tensor格式：
    '''
    1.transforms.Compose()将各种预处理操作组合到一起
    2.transform.ToTensor()将图片转换成 PyTorch 中处理的对象 Tensor.在转化的过程中 PyTorch 自动将图片标准化了，也就是说Tensor的范用是(0,1)之间
    3.transforms.Normalize()要传入两个参数:均值、方差，做的处理就是减均值，再除以方差。将图片转化到了(-1,1)之间
    4.注意因为图片是灰度图，所以只有一个通道，如果是彩色图片，有三个通道，transforms.Normalize([a,b,c],[d,e,f])来表示每个通道对应的均值和方差。
    '''
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # PyTorch 的内置函数 torchvision.datasets.MN工ST 导入数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=data_tf)

    # 注意：测试集为如下形式，评估的准确率为5.0，损失也很高。为什么：
    # test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # torch.utils.data.DataLoader 建立一个数据迭代器，传入数据集和 batch size, 通过 shuffle=True 来表示每次迭代数据的时候是否将数据打乱。
    # 测试集无需打乱顺序；训练集打乱顺序，为了增加训练模型的泛化能力
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数和优化函数
    # 输入的图片尺寸为28*28；两个隐层分别为300和100；最后输出维度是10，因为是10分类问题。
    model = Batch_Net(28 * 28, 300, 100, 10)
    # 可以根据需要选择不同的模型进行分类任务
    # model = Activation_Net(28 * 28, 300, 100, 10)
    # model = Batch_Net(28 * 28, 300, 100, 10)

    # 根据计算机当前硬件环境决定使用cpu还是gpu用于计算
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数为随机梯度下降

    # 训练模型
    epoch = 0
    for data in train_loader:
        img, label = data
        #     print(img.size())  #torch.Size([64, 1, 28, 28])
        img = img.view(img.size(0), -1)  # 将维度变为(64,1*28*28)----(batch,in_dim)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # 此处为训练和迭代的主要部分
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 50 == 0:
            print(f'epoch: {epoch},Train Loss:{loss.data.item():.6f}')

    # 模型评估
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            # with torch.no_grad()表示前向传播时不会保留缓存。测试集不需要做反向传播，所以可以在前向传播时释放掉内存，节约内存空间
            with torch.no_grad():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
        else:
            with torch.no_grad():
                img = Variable(img)
                label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
    print(f'Test Loss:{eval_loss / (len(test_dataset)):.6f},Acc:{eval_acc / (len(test_dataset)):.6f}')


def CNN_model():
    torch.manual_seed(1)  # 为了每次的实验结果一致
    # 设置超参数
    epoches = 2
    batch_size = 50
    learning_rate = 0.001

    # 训练集
    train_data = datasets.MNIST(
        root="./data/",  # 训练数据保存路径
        train=True,  # True为下载训练数据集，False为下载测试数据集
        transform=transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
        download=True,  # 是否需要下载
    )
    # 显示训练集中的第一张图片
    print(train_data.train_data.size())  # [60000,28,28]
    plt.imshow(train_data.train_data[0].numpy())
    plt.show()

    # 测试集
    test_data = datasets.MNIST(root="./data/", train=False)
    print(test_data.test_data.size())  # [10000, 28, 28]
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255
    test_y = test_data.test_labels

    # 将训练数据装入Loader中
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)

    # cnn 实例化
    cnn = CNN_Net()
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)  # batch_x=[50,1,28,28]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 为了实时显示准确率
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

    # 保存训练后的模型
    torch.save(cnn, 'trained_models/cnn_01.pt')
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y)
    print(test_y[:10])


if __name__ == '__main__':
    # Linear_Net_models()
    CNN_model()
