from keras.layers import CategoryEncoding
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
#from monk.monkUtilities import DatasetProcessor  # Importa la classe dal file monkUtilities.py
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split, ParameterSampler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import loguniform 
from cupUtilities import DatasetProcessor
from joblib import parallel_backend
import os
from tqdm import tqdm

def FeatureEngineering():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train, y_train, x_test, y_test = processor.read_tr(split=True)
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20)
    }

    models = {}
    predictions = {}

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        model = SVR()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=30,  # interessante cambiare
            cv=kfold,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )

        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target)
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")

        best_model = rs_cv.best_estimator_
        models[target] = best_model

        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred

        print(f"Metriche per {target}:")
        print(f"MAE: {mean_absolute_error(y_test_target, y_pred)}")
        print(f"MSE: {mean_squared_error(y_test_target, y_pred)}")

    return models, predictions

"""
def FeatureEngineering ():
    # Percorso radice del progetto (due livelli sopra il file corrente)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processor = DatasetProcessor(ROOT_DIR)

    # Carica il dataset di training con split
    x_train, y_train, x_test, y_test = processor.read_tr(split=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    test_data = processor.read_ts()
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_train_poly = poly.fit_transform(x_train)
    X_test_poly = poly.transform(x_test)
    models = {}
    predictions = {}
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20)
    }

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        # Estrai solo la colonna target corrente
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        # Crea un modello SVR
        model = SVR()
        # Creazione dell'istanza di KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # Esegui RandomizedSearchCV per ottimizzare i parametri
        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_jobs=-1,
            cv=kfold,
            scoring="neg_mean_squared_error",  # Cambiato scoring per regressione
            random_state=42
        )
        
        # Fit del modello con barra di progresso
        tqdm.write(f"Optimizing model for {target}...")
        rs_cv.fit(X_train_poly, y_train_target)
        tqdm.write(f"Migliori parametri per {target}: {rs_cv.best_params_}")

        # Salva il modello migliore
        best_model = rs_cv.best_estimator_
        models[target] = best_model

        # Effettua le predizioni
        y_pred = best_model.predict(X_test_poly)
        predictions[target] = y_pred

        # Calcola metriche
        tqdm.write(f"Metriche per {target}:")
        tqdm.write(f"MAE: {mean_absolute_error(y_test_target, y_pred)}")
        tqdm.write(f"MSE: {mean_squared_error(y_test_target, y_pred)}")
"""

def oversamplingSVM (train_path, test_path, results_file):
    #print(Counter(y_train))
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    random_os = RandomOverSampler(random_state = 42)
    X_train, y_train = random_os.fit_resample(X_train, y_train)
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

    results_file.write(f"Migliori parametri AAAAAAAAAAAAAAAA: {rs_cv.best_params_}\n")
    results_file.write(f"Miglior punteggio (cross-validation): {rs_cv.best_score_}\n")

    best_model = rs_cv.best_estimator_
    y_pred = best_model.predict(X_test_encoded)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot()
    plt.show()

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")

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

    best_model = gs_cv.best_estimator_
    y_pred = best_model.predict(X_test_encoded)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

    cm_display.plot()
    plt.show()

    results_file.write("\n Report di classificazione:\n")
    results_file.write(classification_report(y_test, y_pred) + "\n")
    results_file.write(f"Accuratezza sul set di test: {accuracy_score(y_test, y_pred)}\n")



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

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot()
    plt.show()

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

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot()
    plt.show()

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

def nested_grid_search_kfold_with_feature_engineering(train_path, test_path, results_file):
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train, y_train, X_test, y_test = processor.preprocess_data(df_train, df_test)

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train_poly.shape[1]):
        num_tokens = int(np.max(X_train_poly[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train_poly[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test_poly[:, i]).numpy())

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
    FeatureEngineering()


if __name__ == "__main__":
    main()