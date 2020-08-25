#! /usr/bin/env python
#! coding:utf-8:w

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.absolute()


def load_jhmdb_data(
        train_path=current_file_dirpath / Path("../data/JHMDB/GT_train_1.pkl"),
        test_path=current_file_dirpath / Path("../data/JHMDB/GT_test_1.pkl")):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    le = preprocessing.LabelEncoder()
    le.fit(Train['label'])
    print("Loading JHMDB Dataset")
    return Train, Test, le


class JConfig():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 15  # the number of joints
        self.joint_d = 2  # the dimension of joints
        self.clc_num = 21  # the number of class
        self.feat_d = 105
        self.filters = 64

# Genrate dataset
# T: Dataset  C:config le:labelEncoder


def Jdata_generator(T, C, le):
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
