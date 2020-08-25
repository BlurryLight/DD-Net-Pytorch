#! /usr/bin/env python
#! coding:utf-8:w

from sklearn import preprocessing
import pickle
from pathlib import Path
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
