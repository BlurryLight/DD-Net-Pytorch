#! /usr/bin/env python
#! coding:utf-8
from sklearn import preprocessing
import pickle
from tqdm import tqdm
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import sys

Train = pickle.load(open("GT_train_1.pkl", "rb"))
Test = pickle.load(open("GT_test_1.pkl", "rb"))

le = preprocessing.LabelEncoder()
print(le.fit(Train['label']))
print(list(Train), len(Train['label']))

# Temple resizing function


# interpolate l frames to target_l frames
def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            # p_new[:, m, n] = medfilt(p_new[:, m, n], 3) # make no sense. p_new is empty.
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l/l)[:target_l]
    return p_new


# Calculate JCD feature
def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)


def get_CG(p, C):
    M = []
    # upper triangle index with offset 1, which means upper triangle without diagonal
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        # iterate all frames, calc all frame's JCD Matrix
        # p[f].shape (15,2)
        d_m = cdist(p[f], p[f], 'euclidean')
        d_m = d_m[iu]
        # the upper triangle of Matrix and then flattned to a vector. Shape(105)
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)  # normalize
    return M


# Genrate dataset
def data_generator(T, C, le):
    X_0 = []
    X_1 = []
    Y = []
    labels = le.transform(T['label'])
    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i])
        # p.shape (frame,joint_num,joint_coords_dims)
        p = zoom(p, target_l=C.frame_l,
                 joints_num=C.joint_n, joints_dim=C.joint_d)
        # p.shape (target_frame,joint_num,joint_coords_dims)
        # label = np.zeros(C.clc_num)
        # label[labels[i]] = 1
        label = labels[i]
        # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
        M = get_CG(p, C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    return X_0, X_1, Y


class Config():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 15  # the number of joints
        self.joint_d = 2  # the dimension of joints
        self.clc_num = 21  # the number of class
        self.feat_d = 105
        self.filters = 64


# net


def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]

    # x.shape (batch,joint_dim,channel,joint_num,)
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W),
                      align_corners=False, mode='bilinear')
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x


def poses_motion(P):
    # different from the original version
    # TODO: check the funtion, make sure it's right
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_fast = P[:, ::2, :, :]
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    # return (B,target_l,joint_d * joint_n) , (B,target_l/2,joint_d * joint_n)
    return P_diff_slow, P_diff_fast


# poses_motion(torch.from_numpy(X_1), C.frame_l)

# def c1D(x, filters, kernel):
#     x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     return x

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output

# def d1D(x, filters):
#     x = Dense(filters, use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     return x


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output


class DDNet_Original(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(DDNet_Original, self).__init__()
        # JCD part
        self.jcd_conv1 = c1D(frame_l, feat_d, 2 * filters, 1)
        self.jcd_conv2 = c1D(frame_l, 2 * filters, filters, 3)
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.MaxPool1d(kernel_size=2)

        # diff_slow part
        self.slow_conv1 = c1D(frame_l, joint_n * joint_d, 2 * filters, 1)
        self.slow_conv2 = c1D(frame_l, 2 * filters, filters, 3)
        self.slow_conv3 = c1D(frame_l, filters, filters, 1)
        self.slow_pool = nn.MaxPool1d(kernel_size=2)

        # fast_part
        self.fast_conv1 = c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1)
        self.fast_conv2 = c1D(frame_l//2, 2 * filters, filters, 3)
        self.fast_conv3 = c1D(frame_l//2, filters, filters, 1)

        # after cat
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.MaxPool1d(kernel_size=2)

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.MaxPool1d(kernel_size=2)

        self.block3 = block(frame_l//8, 4 * filters, 8 * filters, 3)

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P=None):
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        # pool will downsample the D dim of (B,C,D)
        # but we want to downsample the C channels
        # 1x1 conv may be a better choice
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)
        # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

        x = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# c1 = DDNet_Original(C.frame_l, C.joint_n, C.joint_d,
#                     C.feat_d, C.filters, C.clc_num)
# c1(
#     torch.from_numpy(X_0).type('torch.FloatTensor'),
#     torch.from_numpy(X_1).type('torch.FloatTensor')
# )


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(M, P)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


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
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

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
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)
    C = Config()
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

    Net = DDNet_Original(C.frame_l, C.joint_n, C.joint_d,
                         C.feat_d, C.filters, C.clc_num)
    model = Net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
