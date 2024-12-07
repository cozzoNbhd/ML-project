# This is a sample Python script.

import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.stats import uniform


def load_dataset(train_path, test_path):
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

def preprocess_data(df_train, df_test):
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

def random_grid_search():

    # Caricamento e pre-elaborazione dei dati
    df_train, df_test = load_dataset('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test')
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)

    # Definizione del modello e dei parametri
    model = SVC()
    param_distributions = {
        'kernel': ['linear', 'rbf'],
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1)
    }

    # Implementazione della cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rs_cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_jobs=-1,
        cv=kfold,
        scoring="accuracy",
        random_state=42,
        return_train_score=True
    )

    # Addestramento del modello
    rs_cv.fit(X_train, y_train)

    # Migliori parametri
    print("Migliori parametri: ", rs_cv.best_params_)
    print("Miglior punteggio (cross-validation): ", rs_cv.best_score_)

    # Valutazione del modello sul set di test
    best_model = rs_cv.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))

def grid_search():

    # Caricamento e pre-elaborazione dei dati
    df_train, df_test = load_dataset('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test')
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)

    parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 5, 10], "gamma":[0.001, 0.01, 0.1, 1]}
    #x_train_n=normalize(x_train_list, norm="l2")
    # Implementiamo la cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Create a kernel support vector machine model
    model = SVC()
    gs_cv = GridSearchCV(estimator=model,
            param_grid= parameters,
            cv=kfold,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score = True,
            verbose = 1)

    # Stampiamo i parametri selezionati

    # Addestramento del modello
    gs_cv.fit(X_train, y_train)

    param = gs_cv.cv_results_
    print(param)
    # Train the model on the training data
    #
    #X_train, X_val, y_train, y_val = train_test_split(
    #x_train_list, y_train_list, test_size=0.2, random_state=42)
    #ksvm.fit(x_train_list, y_train_list)
    # Evaluate the model on the test data
    print("Migliori parametri: ", gs_cv.best_params_)
    print("Miglior punteggio (cross-validation): ", gs_cv.best_score_)

    # Addestriamo con i migliori parametri
    best_model = gs_cv.best_estimator_
    best_model.fit(X_train, y_train)

    # Predici sul set di test
    y_pred = best_model.predict(X_test)

    # Valutazione modello
    print("\n Report di classificazione:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))

def nested_grid_search_kfold():

    # Caricamento dei dataset
    df_train, df_test = load_dataset('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test')

    # Pre-elaborazione dei dataset
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)

    # Configurazione dei parametri della Grid Search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    # Definizione dei K-Fold
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Ciclo esterno
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Ciclo interno

    # Modello base
    model = SVC()

    # Memorizzazione dei risultati
    nested_scores = []
    best_params_list = []

    # Ciclo esterno
    for train_idx, val_idx in outer_cv.split(X_train, y_train):
        # Suddivisione dei dati per l'outer loop
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Grid Search nel ciclo interno
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_fold, y_train_fold)

        # Migliori parametri trovati
        best_params = grid_search.best_params_
        best_params_list.append(best_params)

        # Valutazione sul validation set del ciclo esterno
        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val_fold, y_val_fold)
        nested_scores.append(val_score)

    # Risultati finali
    print("\nNested CV Validation Scores:", nested_scores)
    print("Media Nested CV Validation Score:", np.mean(nested_scores))
    print("Migliori parametri trovati nei fold:", best_params_list)

    # Selezione del modello finale
    final_params = best_params_list[np.argmax(nested_scores)]
    print("\nParametri finali selezionati:", final_params)

    # Addestramento del modello finale su tutto il training set
    final_model = SVC(**final_params)
    final_model.fit(X_train, y_train)

    # Valutazione sul test set
    y_test_pred = final_model.predict(X_test)
    print("\nReport di classificazione sul Test Set:")
    print(classification_report(y_test, y_test_pred))
    print("Accuratezza sul Test Set:", accuracy_score(y_test, y_test_pred))

def main():

    print("\n--- Esecuzione di Grid Search ---")
    grid_search()
    print("\n" + "-" * 50)

    print("\n--- Esecuzione di Randomized Grid Search ---")
    random_grid_search()
    print("\n" + "-" * 50)

    print("\n--- Esecuzione di Nested Grid Search ---")
    nested_grid_search_kfold()
    print("\n" + "-" * 50)

# Esegui la funzione main
if __name__ == "__main__":
    main()