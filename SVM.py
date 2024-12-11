import numpy

from utilities import DatasetProcessor  # Importa la classe dal file utilities.py

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

def random_grid_search(train_path, test_path):
    # Crea un'istanza di DatasetProcessor
    processor = DatasetProcessor()

    # Caricamento e pre-elaborazione dei dati
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

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

def grid_search(train_path, test_path):
    # Crea un'istanza di DatasetProcessor
    processor = DatasetProcessor()

    # Caricamento e pre-elaborazione dei dati
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 5, 10], "gamma": [0.001, 0.01, 0.1, 1]}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = SVC()
    gs_cv = GridSearchCV(estimator=model,
                         param_grid=parameters,
                         cv=kfold,
                         scoring="accuracy",
                         n_jobs=-1,
                         return_train_score=True,
                         verbose=1)

    # Addestramento del modello
    gs_cv.fit(X_train, y_train)

    # Risultati e valutazione
    print("Migliori parametri: ", gs_cv.best_params_)
    print("Miglior punteggio (cross-validation): ", gs_cv.best_score_)

    best_model = gs_cv.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n Report di classificazione:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))

def nested_grid_search_kfold(train_path, test_path):
    # Crea un'istanza di DatasetProcessor
    processor = DatasetProcessor()

    # Caricamento dei dataset
    df_train, df_test = processor.load_dataset(train_path, test_path)

    # Pre-elaborazione dei dataset
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Configurazione dei parametri della Grid Search
    param_grid = {
        'C': numpy.linspace(0.001, 10, num=20),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': numpy.linspace(0.001, 1, num=20),
        'degree': [2, 3]
    }

    # Definizione dei K-Fold
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Modello base
    model = SVC()

    nested_scores = []
    param_scores = {}

    for train_idx, val_idx in outer_cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_fold, y_train_fold)

        best_params = grid_search.best_params_

        # Memorizza i parametri e i punteggi
        if tuple(best_params.items()) not in param_scores:
            param_scores[tuple(best_params.items())] = []
        param_scores[tuple(best_params.items())].append(grid_search.best_score_)

        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val_fold, y_val_fold)
        nested_scores.append(val_score)

    print("\nNested CV Validation Scores:", nested_scores)
    print("Media Nested CV Validation Score:", np.mean(nested_scores))

    # Calcola la media dei punteggi per ogni combinazione di parametri
    mean_scores = {params: np.mean(scores) for params, scores in param_scores.items()}
    print("Punteggi medi per combinazione di parametri:", mean_scores)

    # Seleziona i parametri con il punteggio medio più alto
    final_params = max(mean_scores, key=mean_scores.get)
    final_params_dict = dict(final_params)
    print("\nParametri finali selezionati:", final_params_dict)

    # Addestramento del modello finale con i parametri selezionati
    final_model = SVC(**final_params_dict)
    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)
    print("\nReport di classificazione sul Test Set:")
    print(classification_report(y_test, y_test_pred))
    print("Accuratezza sul Test Set:", accuracy_score(y_test, y_test_pred))

def main():
    datasets = [
        ('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        ('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]

    for train_path, test_path in datasets:
        print(f"\n--- Esecuzione di Grid Search per il dataset {train_path} ---")
        grid_search(train_path, test_path)
        print("\n" + "-" * 50)

        print(f"\n--- Esecuzione di Randomized Grid Search per il dataset {train_path} ---")
        random_grid_search(train_path, test_path)
        print("\n" + "-" * 50)

        print(f"\n--- Esecuzione di Nested Grid Search per il dataset {train_path} ---")
        nested_grid_search_kfold(train_path, test_path)
        print("\n" + "-" * 50)

# Esegui la funzione main
if __name__ == "__main__":
    main()
