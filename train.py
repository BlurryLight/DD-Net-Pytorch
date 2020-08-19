#! /usr/bin/env python
#! coding:utf-8
from pickle import FLOAT
from sklearn import preprocessing
import pickle
import sklearn
from torch import scalar_tensor
from tqdm import tqdm
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        label = np.zeros(C.clc_num)
        label[labels[i]] = 1
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


C = Config()
X_0, X_1, Y = data_generator(Train, C, le)
print(X_0.shape)
print(X_1.shape)

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


def poses_motion(P, frame_l):
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_fast = P[:, ::2, :, :]
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    # print(P_diff_slow.shape, P_diff_fast.shape)
    return P_diff_slow, P_diff_fast


# poses_motion(torch.from_numpy(X_1), C.frame_l)

# def c1D(x, filters, kernel):
#     x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     return x

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, 2 * filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,2 * filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,2 * filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, 2 * filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class DDNet(nn.Module):


c1 = block(32, 105, 64, 3)
print(c1(torch.from_numpy(X_0).type('torch.FloatTensor')).shape)

# def block(x, filters):
#     x = c1D(x, filters, 3)
#     x = c1D(x, filters, 3)
#     return x


# def d1D(x, filters):
#     x = Dense(filters, use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     return x


# def build_FM(frame_l=32, joint_n=22, joint_d=2, feat_d=231, filters=16):
#     M = Input(shape=(frame_l, feat_d))
#     P = Input(shape=(frame_l, joint_n, joint_d))

#     diff_slow, diff_fast = pose_motion(P, frame_l)

#     x = c1D(M, filters*2, 1)
#     x = SpatialDropout1D(0.1)(x)
#     x = c1D(x, filters, 3)
#     x = SpatialDropout1D(0.1)(x)
#     x = c1D(x, filters, 1)
#     x = MaxPooling1D(2)(x)
#     x = SpatialDropout1D(0.1)(x)

#     x_d_slow = c1D(diff_slow, filters*2, 1)
#     x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
#     x_d_slow = c1D(x_d_slow, filters, 3)
#     x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
#     x_d_slow = c1D(x_d_slow, filters, 1)
#     x_d_slow = MaxPool1D(2)(x_d_slow)
#     x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

#     x_d_fast = c1D(diff_fast, filters*2, 1)
#     x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
#     x_d_fast = c1D(x_d_fast, filters, 3)
#     x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
#     x_d_fast = c1D(x_d_fast, filters, 1)
#     x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

#     x = concatenate([x, x_d_slow, x_d_fast])
#     x = block(x, filters*2)
#     x = MaxPool1D(2)(x)
#     x = SpatialDropout1D(0.1)(x)

#     x = block(x, filters*4)
#     x = MaxPool1D(2)(x)
#     x = SpatialDropout1D(0.1)(x)

#     x = block(x, filters*8)
#     x = SpatialDropout1D(0.1)(x)

#     return Model(inputs=[M, P], outputs=x)


# def build_DD_Net(C):
#     M = Input(name='M', shape=(C.frame_l, C.feat_d))
#     P = Input(name='P', shape=(C.frame_l, C.joint_n, C.joint_d))

#     FM = build_FM(C.frame_l, C.joint_n, C.joint_d, C.feat_d, C.filters)

#     x = FM([M, P])

#     x = GlobalMaxPool1D()(x)

#     x = d1D(x, 128)
#     x = Dropout(0.5)(x)
#     x = d1D(x, 128)
#     x = Dropout(0.5)(x)
#     x = Dense(C.clc_num, activation='softmax')(x)

#     # Self-supervised part
#     model = Model(inputs=[M, P], outputs=x)
#     return model
