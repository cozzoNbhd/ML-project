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
import matplotlib.pyplot as plt

def main():
    datasets = [
        ('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        ('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]

    # Apri il file in modalità scrittura
    with open("SVM_results.txt", "w") as results_file:
        for train_path, test_path in datasets:
            results_file.write(f"\n--- Esecuzione di Grid Search per il dataset {train_path} ---\n")
            grid_search(train_path, test_path, results_file)
            results_file.write("\n" + "-" * 50 + "\n")

            results_file.write(f"\n--- Esecuzione di Randomized Grid Search per il dataset {train_path} ---\n")
            random_grid_search(train_path, test_path, results_file)
            results_file.write("\n" + "-" * 50 + "\n")

            results_file.write(f"\n--- Esecuzione di Nested Grid Search per il dataset {train_path} ---\n")
            nested_grid_search_kfold(train_path, test_path, results_file)
            results_file.write("\n" + "-" * 50 + "\n")

def grid_search(train_path, test_path, results_file):
    processor = DatasetProcessor()
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

    gs_cv.fit(X_train, y_train)

    results_file.write(f"Migliori parametri: {gs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {gs_cv.best_score_}\n")

    best_model = gs_cv.best_estimator_
    y_pred = best_model.predict(X_test)

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")

    # Plot della curva di apprendimento
    plot_learning_curve(parameters, gs_cv.cv_results_, "Grid Search Learning Curve")

def plot_learning_curve(parameters, cv_results, title):
    for gamma in parameters['gamma']:
        plt.figure()
        c_values = parameters['C']
        test_errors = []

        for c in c_values:
            mean_test_score = [cv_results['mean_test_score'][i] for i in range(len(cv_results['params']))
                               if cv_results['params'][i]['gamma'] == gamma and cv_results['params'][i]['C'] == c]
            test_errors.append(1 - mean_test_score[0])

        plt.plot(c_values, test_errors, marker='o', label=f'Gamma={gamma}')
        plt.title(title)
        plt.xlabel('C')
        plt.ylabel('Test Error')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.show()

def random_grid_search(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    model = SVC()
    param_distributions = {
        'kernel': ['linear', 'rbf'],
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1)
    }

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

    rs_cv.fit(X_train, y_train)

    results_file.write(f"Migliori parametri: {rs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {rs_cv.best_score_}\n")

    best_model = rs_cv.best_estimator_
    y_pred = best_model.predict(X_test)

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")

def nested_grid_search_kfold(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    param_grid = {
        'C': numpy.linspace(0.001, 10, num=20),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': numpy.linspace(0.001, 1, num=20),
        'degree': [2, 3]
    }

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model = SVC()
    nested_scores = []

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
        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val_fold, y_val_fold)
        nested_scores.append(val_score)

    results_file.write(f"\nNested CV Validation Scores: {nested_scores}\n")
    results_file.write(f"Media Nested CV Validation Score: {np.mean(nested_scores)}\n")

    final_model = SVC(**grid_search.best_params_)
    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)
    results_file.write("\nReport di classificazione sul Test Set:\n")
    results_file.write(classification_report(y_test, y_test_pred) + "\n")
    results_file.write(f"Accuratezza sul Test Set: {accuracy_score(y_test, y_test_pred)}\n")

if __name__ == "__main__":
    main()
