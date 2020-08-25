#! /usr/bin/env python
#! coding:utf-8:w

import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.absolute()


def load_shrec_data(
        train_path=current_file_dirpath / Path("../data/SHREC/train.pkl"),
        test_path=current_file_dirpath / Path("../data/SHREC/test.pkl"),
):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    print("Loading SHREC Dataset")
    dummy = None  # return a dummy to provide a similar interface with JHMDB one
    return Train, Test, None


class SConfig():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.class_coarse_num = 14
        self.class_fine_num = 28
        self.feat_d = 231
        self.filters = 64


class Sdata_generator:
    def __init__(self, label_level='coarse_label'):
        self.label_level = label_level

    # le is None to provide a unified interface with JHMDB datagenerator
    def __call__(self, T, C, le=None):
        X_0 = []
        X_1 = []
        Y = []
        for i in tqdm(range(len(T['pose']))):
            p = np.copy(T['pose'][i].reshape([-1, 22, 3]))
            # p.shape (frame,joint_num,joint_coords_dims)
            p = zoom(p, target_l=C.frame_l,
                     joints_num=C.joint_n, joints_dim=C.joint_d)
            # p.shape (target_frame,joint_num,joint_coords_dims)
            # label = np.zeros(C.clc_num)
            # label[labels[i]] = 1
            label = (T[self.label_level])[i] - 1
            # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
            M = get_CG(p, C)

            X_0.append(M)
            X_1.append(p)
            Y.append(label)

        self.X_0 = np.stack(X_0)
        self.X_1 = np.stack(X_1)
        self.Y = np.stack(Y)
        return self.X_0, self.X_1, self.Y


if __name__ == '__main__':
    Train, _ = load_shrec_data()
    C = SConfig()
    X_0, X_1, Y = Sdata_generator('coarse_label')(Train, C, 'coarse_label')
    print(Y)
    X_0, X_1, Y = Sdata_generator('fine_label')(Train, C, 'fine_label')
    print(Y)
    print("X_0.shape", X_0.shape)
    print("X_1.shape", X_1.shape)
