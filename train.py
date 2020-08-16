#! /usr/bin/env python
#! coding:utf-8
from sklearn import preprocessing
import pickle
import sklearn
from tqdm import tqdm
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
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
