import pandas as pd
import numpy as np

class DatasetProcessor:
    """
    Classe per il caricamento e la pre-elaborazione dei dataset di training e test.
    """

    def load_dataset(self, train_path, test_path):
        """
        Carica i dataset di training e test da file CSV.

        Args:
            train_path (str): Percorso del file di training.
            test_path (str): Percorso del file di test.

        Returns:
            tuple: (DataFrame di training, DataFrame di test)
        """
        df_train = pd.read_csv(train_path, sep=r"\s+", header=None)
        df_test = pd.read_csv(test_path, sep=r"\s+", header=None)
        return df_train, df_test

    def preprocess_data(self, df_train, df_test):
        """
        Pre-elabora i dataset di training e test.

        Args:
            df_train (DataFrame): Dataset di training.
            df_test (DataFrame): Dataset di test.

        Returns:
            tuple: (X_train, y_train, X_test, y_test) pre-elaborati come array numpy.
        """
        # Separazione delle feature e del target
        y_train = df_train.iloc[:, 0]
        y_test = df_test.iloc[:, 0]
        X_train = df_train.iloc[:, 1:]
        X_test = df_test.iloc[:, 1:]

        # Conversione in array numpy
        X_train = X_train.select_dtypes(include=[np.number]).to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.select_dtypes(include=[np.number]).to_numpy()
        y_test = y_test.to_numpy()

        return X_train, y_train, X_test, y_test
