# This is a sample Python script.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
def load_data():

    df = pd.read_csv('./datasets/monk/monks-1.train', delim_whitespace=True, header=None)

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
    # Create a kernel support vector machine model
    ksvm = svm.SVC(kernel='rbf',
                gamma=0.1,
                C=10.0)

    X_train, X_val, y_train, y_val = train_test_split(
    x_train_list, y_train_list, test_size=0.20, random_state=42)
    
    # Train the model on the training data
    ksvm.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    accuracy = ksvm.score(X_val, y_val)
    print('Accuracy:', accuracy)

load_data()


