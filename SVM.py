import numpy
from keras.layers import CategoryEncoding
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utilities import DatasetProcessor  # Importa la classe dal file utilities.py
from sklearn.ensemble import BaggingClassifier
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

def ensamble_SVM1(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train.shape[1]):
        num_tokens = int(np.max(X_train[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)
    X_test_encoded = np.concatenate(encoded_test_list, axis=1)

    # Modello Bagging con SVM
    base_model = SVC()
    bagging_model = BaggingClassifier(base_model, n_estimators=10, random_state=42)

    # Creazione dell'istanza di KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    gs_cv = GridSearchCV(
    estimator=bagging_model,
    param_grid={
        'estimator__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),  # Cambiato da base_estimator__ a estimator__
        'estimator__C': [0.1, 1, 5, 10],
        'estimator__gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 1]
    },
    cv=kfold,  # Passa l'istanza di KFold
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True,
    verbose=1
    )

    gs_cv.fit(X_train_encoded, y_train)

    results_file.write(f"Migliori parametri: {gs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {gs_cv.best_score_}\n")


def grid_search(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train.shape[1]):
        num_tokens = int(np.max(X_train[:, i])+1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)
    X_test_encoded = np.concatenate(encoded_test_list, axis=1)

    parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 5, 10], "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 1]}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = SVC()
    gs_cv = GridSearchCV(estimator=model,
                         param_grid=parameters,
                         cv=kfold,
                         scoring="accuracy",
                         n_jobs=-1,
                         return_train_score=True,
                         verbose=1)

    gs_cv.fit(X_train_encoded, y_train)

    results_file.write(f"Migliori parametri: {gs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {gs_cv.best_score_}\n")

    best_model = gs_cv.best_estimator_
    y_pred = best_model.predict(X_test_encoded)

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")

def random_grid_search(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train.shape[1]):
        num_tokens = int(np.max(X_train[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)
    X_test_encoded = np.concatenate(encoded_test_list, axis=1)

    model = SVC()
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20)
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

    rs_cv.fit(X_train_encoded, y_train)

    results_file.write(f"Migliori parametri: {rs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {rs_cv.best_score_}\n")

    best_model = rs_cv.best_estimator_
    y_pred = best_model.predict(X_test_encoded)

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")

def nested_grid_search_kfold(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train.shape[1]):
        num_tokens = int(np.max(X_train[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)
    X_test_encoded = np.concatenate(encoded_test_list, axis=1)

    param_grid = {
        'C': np.linspace(0.1, 10, num=10),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': np.linspace(0.01, 1, num=5)
    }

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model = SVC()
    nested_scores = []
    best_params_list = []

    fig, axs = plt.subplots(5, 1, figsize=(10, 30))
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_encoded, y_train), start=1):
        X_train_fold, X_val_fold = X_train_encoded[train_idx], X_train_encoded[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_fold, y_train_fold)

        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val_fold, y_val_fold)
        nested_scores.append(val_score)

        results = grid_search.cv_results_
        gamma_values = np.array(results['param_gamma'], dtype=float)
        mean_test_scores = np.array(results['mean_test_score'])

        ax = axs[fold_idx - 1]
        ax.plot(gamma_values, mean_test_scores, marker='o', linestyle='-', color='b')
        ax.set_title(f"Fold {fold_idx}: Validation Accuracy for Different Gamma Values")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Validation Accuracy")
        ax.set_xscale("log")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    results_file.write(f"\nNested CV Validation Scores: {nested_scores}\n")
    results_file.write(f"Mean Nested CV Validation Score: {np.mean(nested_scores)}\n")
    results_file.write(f"Best Parameters Across Folds: {best_params_list}\n")

    # Selezione del modello finale
    final_params = best_params_list[np.argmax(nested_scores)]
    print("\nParametri finali selezionati:", final_params)

    final_model = SVC(**grid_search.best_params_)
    final_model.fit(X_train_encoded, y_train)
    y_test_pred = final_model.predict(X_test_encoded)
    results_file.write("\nClassification Report on Test Set:\n")
    results_file.write(classification_report(y_test, y_test_pred) + "\n")
    results_file.write(f"Test Set Accuracy: {accuracy_score(y_test, y_test_pred)}\n")

    # addestramento del modello con feature ridotti DA CONTROLLARE
    # Riduci i dati a 2 dimensioni per il plot
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(X_train_encoded)
    x_test_pca = pca.transform(X_test_encoded)

    # Addestra il modello sull'intero dataset (6 feature originali)
    final_model = SVC(**final_params)
    final_model.fit(x_train_pca, y_train)

    # Genera una griglia nello spazio ridotto (2D PCA)
    x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
    y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predici valori per la griglia (proiezione inversa PCA)
    # grid_pca = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    Z = final_model.predict(grid_pca)
    Z = Z.reshape(xx.shape)

    # Plot della griglia e dei punti
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.rainbow)
    plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, edgecolors='k', s=50, cmap=plt.cm.coolwarm)

    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title(f'SVC con kernel {final_params["kernel"]} e gamma={final_params.get("gamma", "auto")}')
    plt.show()


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

            results_file.write(f"\n--- Esecuzione di Ensamble(bagging) per il dataset {train_path} ---\n")
            ensamble_SVM1(train_path, test_path, results_file)
            results_file.write("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()