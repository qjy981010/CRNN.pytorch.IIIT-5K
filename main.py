#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import argparse
import torch.optim as optim
from warpctc_pytorch import CTCLoss

from crnn import CRNN
from utils import *


def train(root, start_epoch, epoch_num, letters,
          net=None, lr=0.1, fix_width=True):
    """
    Train CRNN model

    Args:
        root (str): Root directory of dataset
        start_epoch (int): Epoch number to start
        epoch_num (int): Epoch number to train
        letters (str): Letters contained in the data
        net (CRNN, optional): CRNN model (default: None)
        lr (float, optional): Coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        fix_width (bool, optional): Scale images to fixed size (default: True)

    Returns:
        CRNN: Trained CRNN model
    """

    # load data
    trainloader = load_data(root, training=True, fix_width=fix_width)
    if not net:
        # create a new model if net is None
        net = CRNN(1, len(letters) + 1)
    # loss function
    criterion = CTCLoss()
    # Adadelta
    optimizer = optim.Adadelta(net.parameters(), lr=lr, weight_decay=1e-3)
    # use gpu or not
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        net = net.to(device)
        criterion = criterion.to(device)
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    # .train() has any effect on Dropout and BatchNorm.
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('----    epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        for i, (img, label) in enumerate(trainloader):
            label, label_length = labeltransformer.encode(label)
            img = img.to(device)
            optimizer.zero_grad()
            # put images in
            outputs = net(img)
            output_length = torch.IntTensor(
                [outputs.size(0)]*outputs.size(1))
            # calc loss
            loss = criterion(outputs, label, output_length, label_length)
            # update
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return net


def test(root, net, letters, fix_width=True):
    """
    Test CRNN model

    Args:
        root (str): Root directory of dataset
        letters (str): Letters contained in the data
        net (CRNN, optional): trained CRNN model
        fix_width (bool, optional): Scale images to fixed size (default: True)
    """

    # load data
    trainloader = load_data(root, training=True, fix_width=fix_width)
    testloader = load_data(root, training=False, fix_width=fix_width)
    # use gpu or not
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        net = net.to(device)
    else:
        print("*****   Warning: Cuda isn't available!  *****")
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')
    # .eval() has any effect on Dropout and BatchNorm.
    net.eval()
    acc = []
    for loader in (testloader, trainloader):
        correct = 0
        total = 0
        for i, (img, origin_label) in enumerate(loader):
            img = img.to(device)

            outputs = net(img)  # length × batch × num_letters
            outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
            outputs = labeltransformer.decode(outputs.data)
            correct += sum([out == real for out,
                            real in zip(outputs, origin_label)])
            total += len(origin_label)
        # calc accuracy
        acc.append(correct / total * 100)
    print('testing accuracy: ', acc[0], '%')
    print('training accuracy: ', acc[1], '%')


def main(epoch_num, lr=0.1, training=True, fix_width=True):
    """
    Main

    Args:
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        fix_width (bool, optional): Scale images to fixed size (default: True)
    """

    model_path = ('fix_width_' if fix_width else '') + 'crnn.pth'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    root = 'data/IIIT5K/'
    if training:
        net = CRNN(1, len(letters) + 1)
        start_epoch = 0
        # if there is pre-trained model, load it
        if os.path.exists(model_path):
            print('Pre-trained model detected.\nLoading model...')
            net.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            print('GPU detected.')
        net = train(root, start_epoch, epoch_num, letters,
                    net=net, lr=lr, fix_width=fix_width)
        # save the trained model for training again
        torch.save(net.state_dict(), model_path)
        # test
        test(root, net, letters, fix_width=fix_width)
    else:
        net = CRNN(1, len(letters) + 1)
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))
        test(root, net, letters, fix_width=fix_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=20, help='number of epochs to train for (default=20)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optim (default=0.1)')
    parser.add_argument('--test', action='store_true', help='Whether to test directly (default is training)')
    parser.add_argument('--fix_width', action='store_true', help='Whether to resize images to the fixed width (default is True)')
    opt = parser.parse_args()
    print(opt)
    main(opt.epoch_num, lr=opt.lr, training=(not opt.test), fix_width=opt.fix_width)