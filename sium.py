# This is a sample Python script.

import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data():

    df = pd.read_csv('./datasets/monk/monks-1.train', sep="\s+", header=None)

    # Controlla la struttura del DataFrame
    #print("Dimensioni del DataFrame originale:", df.shape)

    y_train = df.iloc[:, 0]

    x_train = df.iloc[:, 1:]

    #print(df.head())
    #print(df.columns)
    #print(df.dtypes)

    # Converti in liste di liste
    x_train_list = x_train.select_dtypes(include=[np.number]).to_numpy()
    y_train_list = y_train.to_numpy()

    # Stampa i risultati
    #print("X (lista di liste):")
    #print(x_train_list)

    #print("Y (lista):")
    #print(y_train_list)
    scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(x_train_list)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    x_train_n=normalize(x_train_list, norm="l2")
    #print(standardized_data)
    # Create a kernel support vector machine model
    abc = svm.SVC()
    gs_cv = GridSearchCV(abc, parameters)
    
    gs_cv.fit(x_train_list, y_train_list)
    # Train the model on the training data
    #
    X_train, X_val, y_train, y_val = train_test_split(
    x_train_list, y_train_list, test_size=0.2, random_state=42)
    #ksvm.fit(x_train_list, y_train_list)
    # Evaluate the model on the test data
    accuracy = gs_cv.score(X_val, y_val)
    print('Accuracy:', accuracy)

load_data()


