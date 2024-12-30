import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetProcessor2:
    """
    Classe per il caricamento e la pre-elaborazione dei dataset di training e test.
    """

    def load_dataset(self, train_path, test_path):

        df_train = pd.read_csv(train_path, sep=r"\s+", header=None)
        df_test = pd.read_csv(test_path, sep=r"\s+", header=None)
        return df_train, df_test

    def preprocess_data(self, df_train, df_test):

        # Separazione delle feature e del target
        Y = df_train.iloc[:, -3:]
        X = df_train.iloc[:, 2:-3]
        
        X_ts = df_test.iloc[:,2:-3]

        # Conversione in array numpy
        #X_train = X_train.select_dtypes(include=[np.number]).to_numpy()
        #y_train = Y_train.to_numpy()

        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size= 0.15, random_state=42)

        return X_train, Y_train, X_test, Y_test

    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
