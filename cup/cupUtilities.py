import os

import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
Team_name="Martana"
BLIND_TEST_FILENAME =f"{Team_name}ML-CUP24-TS.csv"
import matplotlib.pyplot as plt
Team_name = "Sium"
BLIND_TEST_FILENAME = f"{Team_name}_ML-CUP24-TS.csv"


class DatasetProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def read_tr(self, split=False):
        file = os.path.join(self.root_dir, "datasets", "ml-cup", "ML-CUP24-TR.csv")
        train = np.loadtxt(file, delimiter=',', usecols=range(1, 16), dtype=np.float64)

        x = train[:, :-3]
        y = train[:, -3:]
        if split:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, y_train, x_test, y_test
        else:
            return x, y

    # Lettura blind test
    def read_ts(self):
        file = os.path.join(self.root_dir, "datasets", "ml-cup", "ML-CUP24-TS.csv")
        test = np.loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)
        return test


    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def write_blind_results(self,y_pred):

        #assert len(y_pred) == 472, "Not enough data were predicted! 472 predictions expected!"

        file = os.path.join(self.root_dir, BLIND_TEST_FILENAME)
        with open(file, "w") as f:
            print("# Tank \t Leo", file=f)
            print("# Martana", file=f)
            print("# ML-CUP24", file=f)
            print("# 02/01/2024", file=f)

            pred_id = 1
            for p in y_pred:
                print("{},{},{}, {}".format(pred_id, p[0], p[1], p[2]), file=f)
                print("{},{},{},{}".format(pred_id, p[0], p[1], p[2]), file=f)
                pred_id += 1

        f.close()

    def save_figure(self, model_name, **params):
        name = ""
        for k, v in params.items():
            name += f"{k}{v}_"
        name += ".png"

        # create plot directory if it doesn't exist
        dir_path = os.path.join(self.root_dir, model_name, "plot")
        os.makedirs(dir_path, exist_ok=True)

        # save plot as figure
        fig_path = os.path.join(dir_path, name)
        plt.savefig(fig_path, dpi=600)


