#! /usr/bin/env python
#! coding:utf-8
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.jhmdb_loader import load_jhmdb_data, Jdata_generator, JConfig
from dataloader.shrec_loader import load_shrec_data, Sdata_generator, SConfig
from models.DDNet_Original import DDNet_Original as DDNet
from utils import makedir
import sys
import time
sys.path.insert(0, './pytorch-summary/torchsummary/')
from torchsummary import summary  # noqa

history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(M, P)
        loss = criterion(output, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    history['train_loss'].append(train_loss)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # output shape (B,Class)
            # target_shape (B)
            # pred shape (B,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train (default: 199)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=int, required=True, metavar='N',
                        help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine, others is undefined')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)

    # alias
    Config = None
    data_generator = None
    load_data = None
    clc_num = 0
    if args.dataset == 0:
        Config = JConfig()
        data_generator = Jdata_generator
        load_data = load_jhmdb_data
        clc_num = Config.clc_num
    elif args.dataset == 1:
        Config = SConfig()
        load_data = load_shrec_data
        clc_num = Config.class_coarse_num
        data_generator = Sdata_generator('coarse_label')
    elif args.dataset == 2:
        Config = SConfig()
        clc_num = Config.class_fine_num
        load_data = load_shrec_data
        data_generator = Sdata_generator('fine_label')
    else:
        print("Unsupported dataset!")
        sys.exit(1)

    C = Config
    Train, Test, le = load_data()
    X_0, X_1, Y = data_generator(Train, C, le)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t, X_1_t, Y_t = data_generator(Test, C, le)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    trainset = torch.utils.data.TensorDataset(X_0, X_1, Y)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size)

    Net = DDNet(C.frame_l, C.joint_n, C.joint_d,
                C.feat_d, C.filters, clc_num)
    model = Net.to(device)

    summary(model, [(C.frame_l, C.feat_d), (C.frame_l, C.joint_n, C.joint_d)])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, factor=args.gamma, patience=5, cooldown=0.5, min_lr=5e-6, verbose=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step(train_loss)

    savedir = Path('experiments') / Path(str(int(time.time())))
    makedir(savedir)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(history['train_loss'])
    ax1.plot(history['test_loss'])
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')

    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_acc'])
    fig.tight_layout()
    fig.savefig(str(savedir / "temp.png"))
    if args.save_model:
        torch.save(model.state_dict(), str(savedir/"model.pt"))


if __name__ == '__main__':
    main()
