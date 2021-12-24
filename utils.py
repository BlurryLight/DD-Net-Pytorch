#! /usr/bin/env python
#! coding:utf-8

import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pathlib
import copy
from scipy.signal import medfilt

# Temple resizing function
# interpolate l frames to target_l frames


def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            # p_new[:, m, n] = medfilt(p_new[:, m, n], 3) # make no sense. p_new is empty.
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p_copy[:, m, n], target_l/l)[:target_l]
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


def makedir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
