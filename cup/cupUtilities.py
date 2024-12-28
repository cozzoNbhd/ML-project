import os

import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def read_tr(self, split=False):
        file = os.path.join(self.root_dir, "datasets", "ml-cup", "ML-CUP24-TR.csv")
        train = np.loadtxt(file, delimiter=',', usecols=range(1, 16), dtype=np.float64)

        x = train[:, :-3]
        y = train[:, -3:]
        if split:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
            return x_train, y_train, x_test, y_test
        else:
            return x, y

    def read_ts(self):
        file = os.path.join(self.root_dir, "datasets", "ml-cup", "ML-CUP24-TS.csv")
        test = np.loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)
        return test
