#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import pickle
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from warpctc_pytorch import CTCLoss
from PIL import Image
from collections import Iterable


class FixHeightResize(object):
    """
    对图片做固定高度的缩放
    """

    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img 为 PIL.Image 对象
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)


class IIIT5k(Dataset):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False
    """

    def __init__(self, root, training=True, fix_width=False):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        data = sio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in data])

        # 图片缩放 + 转化为灰度图 + 转化为张量
        transform = [transforms.Resize((32, 100), Image.ANTIALIAS)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)

        # 加载图片
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    # 以下两个方法必须要重载
    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


class CRNN(nn.Module):
    """
    CRNN模型

    Args:
        in_channels (int): 输入的通道数，如果是灰度图则为1，如果没有灰度化则为3
        out_channels (int): 输出的通道数（类别数），即样本里共有多少种字符
    """

    def __init__(self, in_channels, out_channels):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 256
        # CNN 结构与参数
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        # 池化层结构
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        # 是否加入批归一化层
        self.batchnorm = (False, False, False, True, False)
        self.cnn = self._get_cnn_layers()
        # RNN 两层双向LSTM。pytorch中LSTM的输出通道数为hidden_size *
        # num_directions,这里因为是双向的，所以num_directions为2
        self.rnn1 = nn.LSTM(self.cnn_struct[-1][-1],
                            hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        # 最后一层全连接
        self.fc = nn.Linear(hidden_size*2, out_channels)
        # 初始化参数，不是很重要
        self._initialize_weights()

    def forward(self, x):   # input: height=32, width>=100
        x = self.cnn(x)   # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)   # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)   # width>=24, batch, channel=512
        x = self.rnn1(x)[0]   # length=width>=24, batch, channel=256*2
        x = self.rnn2(x)[0]   # length=width>=24, batch, channel=256*2
        l, b, h = x.size()
        x = x.view(l*b, h)   # length*batch, hidden_size*2
        x = self.fc(x)   # length*batch, output_size
        x = x.view(l, b, -1)   # length>=24, batch, output_size
        return x

    # 构建CNN层
    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LabelTransformer(object):
    """
    字符编码解码器

    Args:
        letters (str): 所有的字符组成的字符串
    """

    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def encode(self, text):
        if isinstance(text, str):
            length = [len(text)]
            result = [self.encode_map[letter] for letter in text]
        else:
            length = []
            result = []
            for word in text:
                length.append(len(word))
                result.extend([self.encode_map[letter] for letter in word])
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
            result.append(''.join(word))
        return result


def load_data(root, training=True, fix_width=False):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False

    Return:
        加载的训练集或者测试集
    """
    if training:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(
            root, 'train'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=True, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    else:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(
            root, 'test'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=False, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def train(root, start_epoch, epoch_num, letters,
          net=None, lr=0.1, fix_width=False):
    """
    训练CRNN

    Args:
        root (str): 存放数据集的文件夹
        start_epoch (int): 开始训练的是第多少次epoch，便于对训练过程的追踪回顾。
        epoch_num (int): 将训练的epoch数目
        letters (str): 所有的字符组成的字符串
        net (CRNN, optional): 之前训练过的网络
        lr (float, optional): 学习速率，默认为0.1
        fix_width (bool, optional): 是否固定宽度，默认固定

    Returns:
        CRNN: 训练好的模型
    """
    # 加载数据
    trainloader = load_data(root, training=True, fix_width=fix_width)
    # 判断GPU是否可用
    use_cuda = torch.cuda.is_available()
    if not net:
        # 如果没有之前训练好的模型，就新建一个
        net = CRNN(1, len(letters) + 1)
    # 损失函数
    criterion = CTCLoss()
    # 优化方法采用Adadelta
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    # 构建编码解码器
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    # .train() 对批归一化有一定的作用
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('----    epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        for i, (img, label) in enumerate(trainloader):
            label, label_length = labeltransformer.encode(label)
            if use_cuda:
                img = img.cuda()
            img, label = Variable(img), Variable(label)
            label_length = Variable(label_length)
            # 清空梯度
            optimizer.zero_grad()
            # 将图片输入
            outputs = net(img)
            output_length = Variable(torch.IntTensor(
                [outputs.size(0)]*outputs.size(1)))
            # 计算损失
            loss = criterion(outputs, label, output_length, label_length)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return net


def test(root, net, letters, fix_width=True):
    """
    测试CRNN模型

    Args:
        root (str): 存放数据集的文件夹
        letters (str): 所有的字符组成的字符串
        net (CRNN, optional): 训练好的网络
        fix_width (bool, optional): 是否固定宽度，默认固定
    """
    # 加载数据
    testloader = load_data(root, training=False, fix_width=fix_width)
    # 判断GPU是否可用
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    # 构建编码解码器
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')
    # .eval() 对批归一化有一定的作用
    net.eval()
    correct = 0
    for i, (img, origin_label) in enumerate(testloader):
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        outputs = net(img)  # length × batch × num_letters
        outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
        outputs = labeltransformer.decode(outputs.data)
        correct += sum([out == real for out,
                        real in zip(outputs, origin_label)])
    # 计算准确率
    print('test accuracy: ', correct / 30, '%')


def main(training=True, fix_width=False):
    """
    主函数，控制train与test的调用以及模型的加载存储等

    Args:
        training (bool, optional): 为True是训练，为False是测试，默认为True
        fix_width (bool, optional): 是否固定图片宽度，默认为False
    """
    file_name = ('fix_width_' if fix_width else '') + 'crnn_.pkl'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    root = 'data/IIIT5K/'
    if training:
        net = None
        start_epoch = 0
        epoch_num = 2  # 每训练两个epoch进行一次测试
        lr = 0.1
        if os.path.exists(file_name):
            print('Pre-trained model detected.\nLoading model...')
            start_epoch, net = pickle.load(open(file_name, 'rb'))
        if torch.cuda.is_available():
            print('GPU detected.')
        for i in range(5):
            net = train(root, start_epoch, epoch_num, letters,
                        net=net, lr=lr, fix_width=fix_width)
            start_epoch += epoch_num
            test(root, net, letters, fix_width=fix_width)
        # 将训练的epoch数与我们的模型保存起来，模型还可以加载出来继续训练
        pickle.dump((start_epoch, net), open(file_name, 'wb'), True)
    else:
        start_epoch, net = pickle.load(open(file_name, 'rb'))
        test(root, net, letters, fix_width=fix_width)


if __name__ == '__main__':
    main(training=True, fix_width=False)
