#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import pickle
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from crnn import CRNN
from utils import *


def train(root, start_epoch, epoch_num, letters,
          net=None, lr=0.1, fix_width=True):
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


def main(training=True, fix_width=True):
    """
    主函数，控制train与test的调用以及模型的加载存储等

    Args:
        training (bool, optional): 为True是训练，为False是测试，默认为True
        fix_width (bool, optional): 是否固定图片宽度，默认为True
    """

    file_name = ('fix_width_' if fix_width else '') + 'crnn.pkl'
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
    main(training=True, fix_width=True)