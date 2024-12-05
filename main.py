# This is a sample Python script.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def load_data():

    df = pd.read_csv('./datasets/monk/monks-1.train', delim_whitespace=True, header=None)
    print(df)

    # Controlla la struttura del DataFrame
    print("Dimensioni del DataFrame originale:", df.shape)

    y_train = df.iloc[:, 0]

    x_train = df.iloc[:, 1:]

    print(df.head())
    print(df.columns)
    print(df.dtypes)

    # Converti in liste di liste
    x_train_list = x_train.select_dtypes(include=[np.number]).to_numpy()
    y_train_list = y_train.to_numpy()

    # Stampa i risultati
    print("X (lista di liste):")
    print(x_train_list)

    print("Y (lista):")
    print(y_train_list)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x_train_list)
    print(standardized_data)


load_data()


