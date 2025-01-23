from keras.layers import CategoryEncoding
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split, ParameterSampler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, make_scorer
#from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, PowerTransformer
from scipy.stats import loguniform 
from cupUtilities import DatasetProcessor
from cup2Utilities import DatasetProcessor2
from joblib import parallel_backend
from sklearn.multioutput import MultiOutputRegressor
import tensorflow.keras.backend as K
import os
import tensorflow as tf
import xgboost as xgb


from sklearn.multioutput import MultiOutputRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import seaborn as sns



import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Definizione della funzione MEE
def mean_euclidean_error(y_true, y_pred):
    # Calcola la distanza euclidea media
    
    mee = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    
    # Converte il tensor in float (necessario per sklearn)
    return float(mee.numpy())

"""
def random_grid_search():

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    
    # Load and split data
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()
    
    # Normalize features (not targets)
    x_train, x_test = processor.normalize_data(x_train_n, x_test_n)
    x_ts_norm, _ = processor.normalize_data(x_ts, x_train_n)
    
    # Hyperparameter distributions
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20),
        'epsilon': np.random.uniform(0.01, 0.1, 10)
    }
    
    models = {}
    predictions = {}
    final_predictions = []
    cv_figures = {}  # Store figures for each target
    
    # Custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    target_names = ['TARGET_x', 'TARGET_y', 'TARGET_z']
    
    for i, target in enumerate(target_names):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        
        model = SVR()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=100,
            cv=kfold,
            scoring=mee_scorer,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        
        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target)
        print(f"Best parameters for {target}: {rs_cv.best_params_}")
        
        # Create and save visualization
        cv_fig = plot_cv_results(rs_cv, target)
        cv_figures[target] = cv_fig
        
        best_model = rs_cv.best_estimator_
        models[target] = best_model
        
        # Validation predictions
        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred
        
        # Print metrics
        mee_value = mean_euclidean_error(y_test_target, y_pred)
        print(f"MEE for validation set ({target}): {mee_value}")
        print(f"Best model train score: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]:.4f} "
              f"± {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]:.4f}")
        print(f"Best model validation score: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]:.4f} "
              f"± {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]:.4f}")
        
        # Save the visualization
        cv_fig.savefig(f'cv_results_{target}.png')
        plt.close(cv_fig)
        
        # Test set predictions
        pred_test = best_model.predict(x_ts_norm)
        final_predictions.append(pred_test)
    
    # Combine predictions and save results
    final_predictions = np.array(final_predictions).T
    processor.write_blind_results(final_predictions)
    print("Results saved successfully!")
    
    return models, predictions, cv_figures
"""

import os
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt


def random_grid_search_multioutput():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    
    # Load and split data
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()
    
    # Normalize features (not targets)
    x_train, x_test = processor.normalize_data(x_train_n, x_test_n)
    x_ts_norm, _ = processor.normalize_data(x_ts, x_train_n)
    
    # Hyperparameter distributions
    param_distributions = {
        'estimator__kernel': ['linear', 'rbf', 'poly'],
        'estimator__C': np.random.uniform(0.1, 10, 20),
        'estimator__gamma': np.random.uniform(0.01, 1, 20),
        'estimator__epsilon': np.random.uniform(0.01, 0.1, 10)
    }
    
    # Custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    # Multioutput SVR wrapped in MultiOutputRegressor
    base_model = SVR()
    multioutput_model = MultiOutputRegressor(base_model)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Randomized search for hyperparameter tuning
    rs_cv = RandomizedSearchCV(
        estimator=multioutput_model,
        param_distributions=param_distributions,
        n_iter=100,
        cv=kfold,
        scoring=mee_scorer,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    
    print("Optimizing multioutput model...")
    rs_cv.fit(x_train, y_train)
    print(f"Best parameters: {rs_cv.best_params_}")
    
    # Save cross-validation visualization
    cv_fig = plot_cv_results(rs_cv, 'MultiOutput')
    cv_fig.savefig('cv_results_multioutput.png')
    plt.close(cv_fig)
    
    best_model = rs_cv.best_estimator_
    
    # Validation predictions
    y_pred = best_model.predict(x_test)
    mee_value = mean_euclidean_error(y_test, y_pred)
    print(f"MEE for validation set: {mee_value}")
    print(f"Best model train score: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]:.4f} "
          f"± {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]:.4f}")
    print(f"Best model validation score: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]:.4f} "
          f"± {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]:.4f}")
    
    # Test set predictions
    final_predictions = best_model.predict(x_ts_norm)
    
    # Save final predictions
    processor.write_blind_results(final_predictions)
    print("Results saved successfully!")
    
    return best_model, y_pred, cv_fig


def calcola(y_true_scaled, y_pred_scaled, scaler, alpha=0.001):
    """
    Funzione per calcolare la MAE sui valori originali.
    """
    y_true_original = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Calcola MSE come base
    mse = mean_squared_error(y_true_original, y_pred_original)
    
    # Penalizzazione L2
    penalty_L2 = alpha * np.sum(np.square(y_pred_scaled))  # Penalità L2
    
    # La funzione restituisce il negativo della loss (se greater_is_better=True)
    return -(mse + penalty_L2)


def random_grid_search3():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)
    x_ts = p2.normalize_data(x_train_n, x_ts)

    # Parametri per la ricerca casuale
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 15, 200),
        'gamma': np.random.uniform(0.01, 0.5, 200)  # Gamma ridotto
    }

    models = {}
    predictions = {}

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        plt.hist(y_train[:, i], bins=30, alpha=0.7, label=target)
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        # Creazione dello scaler per i target
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_target_scaled = target_scaler.fit_transform(y_train_target.reshape(-1, 1))
        y_test_target_scaled = target_scaler.transform(y_test_target.reshape(-1, 1))

        # Creare lo scorer per il target specifico
        custom_scorer = make_scorer(calcola, greater_is_better=False, scaler=target_scaler, alpha=0.001)

        model = SVR()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=100,
            cv=kfold,
            scoring=custom_scorer,  # Passa lo scorer personalizzato
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target_scaled.ravel())  # Usa i target scalati per il training
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")

        best_model = rs_cv.best_estimator_
        best_params = rs_cv.best_params_
        models[target] = best_model

        # Predizioni con il modello migliore
        y_pred_scaled = best_model.predict(x_test)

        # Inversa la trasformazione dei target predetti per calcolare la loss nell'originale spazio
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Calcola le metriche di errore
        mse = mean_squared_error(y_test_target, y_pred_original)
        print(f"Metriche per {target}:")
        print(f"MSE: {mse}")
        print(f"Train score medio: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]}") 
    plt.legend()
    plt.show()    


#qui applico la standardizzazione anche al target
def random_grid_search2():
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2=DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    x_tn, x_ts = p2.normalize_data(x_train_n, x_ts)

    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20),
        'epsilon': np.random.uniform(0.01, 0.1, 10)
    }

    models = {}
    predictions = {}

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        # Creazione dello scaler per i target (dipende dal tuo dataset, puoi anche usare un altro scaler)
        target_scaler = StandardScaler()
        y_train_target_scaled = target_scaler.fit_transform(y_train_target.reshape(-1, 1))
        y_test_target_scaled = target_scaler.transform(y_test_target.reshape(-1, 1))

        model = SVR()
        kfold = KFold(n_splits=2, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=30,  # cambia questo valore a seconda delle tue necessità
            cv=kfold,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target_scaled.ravel())  # Use the scaled targets for training
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")
        best_model = rs_cv.best_estimator_
        best_params = rs_cv.best_params_
        models[target] = best_model

        # Predizioni con il modello migliore
        y_pred_scaled = best_model.predict(x_test)

        # Inversa la trasformazione dei target predetti per calcolare la loss nell'originale spazio
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Calcola le metriche di errore usando i target originali
        print(f"Metriche per {target}:")
        print(f"Train Score del modello ottimale: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]}")
        print(f"Train standard deviation del modello ottimale: {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]}")

        print(f"Validation Score del modello ottimale: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]}")
        print(f"Validation standard deviation del modello ottimale: {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]}")

        print(f"MSE: {mean_squared_error(y_test_target, y_pred_original)}")

        #result_content = f"Target: {target}\nMigliori parametri: {best_params}\nMAE: {mean_absolute_error(y_test_target, y_pred_original)}\nMSE: {mean_squared_error(y_test_target, y_pred_original)}\n"
        #write_results_to_file('cupSVM_result.txt', result_content)

        
def plot_cv_results(rs_cv, target_name):
    """
    Create visualizations for cross-validation results
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get results
    results = rs_cv.cv_results_
    
    # 1. Distribution of scores
    scores = -results['mean_test_score']  # Negative because we used negative MEE
    sns.histplot(scores, ax=ax1, kde=True)
    ax1.axvline(x=-rs_cv.best_score_, color='r', linestyle='--', label='Best score')
    ax1.set_title(f'Distribution of CV Scores for {target_name}')
    ax1.set_xlabel('Mean Euclidean Error')
    ax1.legend()
    
    # 2. Learning curves for best model
    train_scores = -results['mean_train_score']
    test_scores = -results['mean_test_score']
    sorted_idx = np.argsort(test_scores)
    ax2.plot(train_scores[sorted_idx], label='Train')
    ax2.plot(test_scores[sorted_idx], label='Validation')
    ax2.set_title(f'Learning Curves for {target_name}')
    ax2.set_xlabel('Sorted Trial Index')
    ax2.set_ylabel('Mean Euclidean Error')
    ax2.legend()
    
    # 3. Parameter importance for numerical parameters
    for param in ['C', 'gamma', 'epsilon']:
        param_values = results['param_' + param].data
        sns.scatterplot(x=param_values, y=-results['mean_test_score'], 
                       alpha=0.5, ax=ax3)
    ax3.set_title(f'Parameter Impact on Score for {target_name}')
    ax3.set_ylabel('Mean Euclidean Error')
    ax3.legend(['C', 'gamma', 'epsilon'])
    
    # 4. Kernel comparison boxplot
    kernel_scores = []
    kernel_types = []
    for kernel in ['linear', 'rbf', 'poly']:
        mask = results['param_kernel'].data == kernel
        kernel_scores.extend(-results['mean_test_score'][mask])
        kernel_types.extend([kernel] * sum(mask))
    
    sns.boxplot(x=kernel_types, y=kernel_scores, ax=ax4)
    ax4.set_title(f'Performance by Kernel Type for {target_name}')
    ax4.set_ylabel('Mean Euclidean Error')
    
    plt.tight_layout()
    return fig


def random_grid_search():

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    
    # Load and split data
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()
    
    # Normalize features (not targets)
    x_train, x_test = processor.normalize_data(x_train_n, x_test_n)
    x_ts_norm, _ = processor.normalize_data(x_ts, x_train_n)
    
    # Hyperparameter distributions
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20),
        'epsilon': np.random.uniform(0.01, 0.1, 10)
    }
    
    models = {}
    predictions = {}
    final_predictions = []
    cv_figures = {}  # Store figures for each target
    
    # Custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    target_names = ['TARGET_x', 'TARGET_y', 'TARGET_z']
    
    for i, target in enumerate(target_names):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        
        model = SVR()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=100,
            cv=kfold,
            scoring=mee_scorer,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        
        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target)
        print(f"Best parameters for {target}: {rs_cv.best_params_}")
        
        # Create and save visualization
        cv_fig = plot_cv_results(rs_cv, target)
        cv_figures[target] = cv_fig
        
        best_model = rs_cv.best_estimator_
        models[target] = best_model
        
        # Validation predictions
        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred
        
        # Print metrics
        mee_value = mean_euclidean_error(y_test_target, y_pred)
        print(f"MEE for validation set ({target}): {mee_value}")
        print(f"Best model train score: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]:.4f} "
              f"± {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]:.4f}")
        print(f"Best model validation score: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]:.4f} "
              f"± {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]:.4f}")
        
        # Save the visualization
        cv_fig.savefig(f'cv_results_{target}.png')
        plt.close(cv_fig)
        
        # Test set predictions
        pred_test = best_model.predict(x_ts_norm)
        final_predictions.append(pred_test)
    
    # Combine predictions and save results
    final_predictions = np.array(final_predictions).T
    processor.write_blind_results(final_predictions)
    print("Results saved successfully!")
    
    return models, predictions, cv_figures



def random_grid_search_different_feature_scaling():
    """
    Funzione per ottimizzare i modelli con scaling solo sulle feature.
    I target non vengono scalati. Si confrontano StandardScaler e MinMaxScaler per le feature.
    """
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Configurazione degli scaler per le feature
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(feature_range=(0, 1))
    }

    # Parametri per RandomizedSearchCV
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 10, 20),
        'gamma': np.random.uniform(0.01, 1, 20),
        'epsilon': np.random.uniform(0.01, 0.1, 10)
    }

    for scaler_name, scaler in scalers.items():
        print(f"\n--- Inizio ottimizzazione con {scaler_name} ---\n")

        # Applica lo scaling alle feature
        x_train = scaler.fit_transform(x_train_n)
        x_test = scaler.transform(x_test_n)
        x_ts = scaler.transform(x_ts)

        models = {}
        predictions = {}
        final_predictions = []

        # Itera sui target
        for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
            y_train_target = y_train[:, i]
            y_test_target = y_test[:, i]

            # Configura il modello SVR e RandomizedSearchCV
            model = SVR()
            kfold = KFold(n_splits=2, shuffle=True, random_state=42)

            rs_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=30,
                cv=kfold,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
                return_train_score=True
            )

            print(f"Optimizing {target} with {scaler_name}...")
            rs_cv.fit(x_train, y_train_target)
            print(f"Migliori parametri per {target}: {rs_cv.best_params_}")

            best_model = rs_cv.best_estimator_
            models[target] = best_model

            # Predizioni sul test set
            y_pred = best_model.predict(x_test)
            predictions[target] = y_pred

            # Calcolo metriche di errore usando i target originali
            print(f"Metriche per {target}:")
            print(f"Train Score del modello ottimale: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]}")  
            print(f"Train standard deviation del modello ottimale: {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]}")

            print(f"Validation Score del modello ottimale: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]}")  
            print(f"Validation standard deviation del modello ottimale: {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]}")

            print(f"MSE per il set di validazione: {mean_squared_error(y_test[:, i], y_pred)}")

            # Predizioni finali su x_ts
            pred_test = best_model.predict(x_ts)
            final_predictions.append(pred_test)

        # Combina tutte le predizioni in un array (N, 3)
        final_predictions = np.array(final_predictions).T  # Trasposta per avere (N, 3)



def optimized_random_grid_search():
    """
    Funzione per ottimizzare il modello per ogni target con trasformazioni specifiche
    e una loss personalizzata.
    """
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Normalizzazione delle feature
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    # Configurazione dei parametri per RandomizedSearchCV
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 15, 200),
        'gamma': np.random.uniform(0.01, 2, 200)
    }

    models = {}
    predictions = {}

    # Trasformazioni e pesi specifici per ogni target
    target_settings = {
        'TARGET_x': {
            'scaler': StandardScaler(),
            'model': SVR(),
            'weight': 1.0  # Peso normale
        },
        'TARGET_y': {
            'scaler': StandardScaler(),
            'model': SVR(),
            'weight': 1.0  # Peso normale
        },
        'TARGET_z': {
            'scaler': PowerTransformer(method='yeo-johnson'),
            'model': SVR(),
            'weight': 2.0  # Peso maggiore per la complessità
        }
    }

    # Definizione della funzione di loss personalizzata
    def custom_loss(y_true_scaled, y_pred_scaled, scaler, alpha=0.001, target_weight=1.0):
        """
        Funzione di loss personalizzata con penalità L2 e pesi sui target.
        """
        y_true_original = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        mse = mean_squared_error(y_true_original, y_pred_original)
        penalty_L2 = alpha * np.sum(np.square(y_pred_scaled))
        return -(mse * target_weight + penalty_L2)  # Ritorna il negativo per massimizzare

    # Ottimizzazione per ogni target
    for i, (target, settings) in enumerate(target_settings.items()):
        print(f"Optimizing {target}...")  # Aggiungi un print prima dell'ottimizzazione
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        # Applica lo scaler specifico
        target_scaler = settings['scaler']
        y_train_target_scaled = target_scaler.fit_transform(y_train_target.reshape(-1, 1))
        y_test_target_scaled = target_scaler.transform(y_test_target.reshape(-1, 1))

        # Scorer personalizzato
        custom_scorer = make_scorer(
            custom_loss,
            greater_is_better=True,
            scaler=target_scaler,
            alpha=0.001,
            target_weight=settings['weight']
        )

        model = settings['model']
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=100,
            cv=kfold,
            scoring=custom_scorer,  # Scorer personalizzato
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        # Add Debugging print before fitting
        print("Fitting the model...")
        rs_cv.fit(x_train, y_train_target_scaled.ravel())
        
        best_model = rs_cv.best_estimator_
        models[target] = best_model

        # Predizioni con il modello migliore
        y_pred_scaled = best_model.predict(x_test)

        # Inversa la trasformazione dei target predetti per calcolare la loss nello spazio originale
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Calcola le metriche di errore
        mse = mean_squared_error(y_test_target, y_pred_original)
        train_score = rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]

        # Add print statements to debug
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")
        print(f"Metriche per {target}:")
        print(f"MSE: {mse}")
        print(f"Train score medio: {train_score}")

        predictions[target] = y_pred_original

    return models, predictions


#provata così e fa abbastanza schifo
def optimized_random_grid_search2():
    """
    Funzione per ottimizzare il modello per ogni target con trasformazioni specifiche,
    una loss personalizzata, e pesi sui target.
    """
    
    # Lettura dei dati
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Normalizzazione delle feature
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    # Configurazione dei parametri per RandomizedSearchCV
    param_distributions = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.random.uniform(0.1, 15, 200),
        'gamma': np.random.uniform(0.01, 2, 200)
    }

    models = {}
    predictions = {}

    # Trasformazioni e pesi specifici per ogni target
    target_settings = {
        'TARGET_x': {
            'scaler': StandardScaler(),
            'model': SVR(),
            'weight': 1.0  # Peso normale
        },
        'TARGET_y': {
            'scaler': StandardScaler(),
            'model': SVR(),
            'weight': 1.0  # Peso normale
        },
        'TARGET_z': {
            'scaler': PowerTransformer(method='yeo-johnson'),
            'model': SVR(),
            'weight': 2.0  # Peso maggiore per complessità
        }
    }

    # Definizione della funzione di loss personalizzata
    def custom_loss(y_true_scaled, y_pred_scaled, scaler, alpha=0.001, target_weight=1.0):
        """
        Funzione di loss personalizzata con pesi sui target.
        """
        y_true_original = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        mse = mean_squared_error(y_true_original, y_pred_original)
        return -(mse * target_weight)  # Rimuovi penalità L2

    # Ottimizzazione per ogni target
    for i, (target, settings) in enumerate(target_settings.items()):
        print(f"\nOptimizing {target}...")  # Messaggio di inizio ottimizzazione
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        # Applica lo scaler specifico per il target
        target_scaler = settings['scaler']
        y_train_target_scaled = target_scaler.fit_transform(y_train_target.reshape(-1, 1))
        y_test_target_scaled = target_scaler.transform(y_test_target.reshape(-1, 1))

        # Scorer personalizzato
        custom_scorer = make_scorer(
            custom_loss,
            greater_is_better=True,
            scaler=target_scaler,
            alpha=0.001,
            target_weight=settings['weight']
        )

        model = settings['model']
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=50,
            cv=kfold,
            scoring=custom_scorer,  # Scorer personalizzato
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        # Debug: Messaggio prima del fitting
        print(f"Fitting the model for {target}...")
        rs_cv.fit(x_train, y_train_target_scaled.ravel())
        
        best_model = rs_cv.best_estimator_
        models[target] = best_model

        # Predizioni con il modello migliore
        y_pred_scaled = best_model.predict(x_test)

        # Inversa la trasformazione dei target predetti per calcolare la loss nello spazio originale
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Calcolo delle metriche di errore
        mse = mean_squared_error(y_test_target, y_pred_original)
        train_score = rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]

        # Debug: Stampa dei risultati
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")
        print(f"Metriche per {target}:")
        print(f"MSE: {mse}")
        print(f"Train score medio: {train_score}")

        predictions[target] = y_pred_original

    return models, predictions


#grid search
def grid_search():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2=DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    x_tn, x_ts = p2.normalize_data(x_train_n, x_ts)
    models = {}
    predictions = {}
    # Creazione dello scorer personalizzato (valori negativi per coerenza con minimizzazione della loss)
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 5, 10], "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 1], 'epsilon': [0.001, 0.01, 0.1, 0.5, 1]}
        kfold = KFold(n_splits=2, shuffle=True, random_state=42)
        model = SVR()
        gs_cv = GridSearchCV(estimator=model,
                            param_grid=parameters,
                            cv=kfold,
                            scoring=mee_scorer,
                            n_jobs=-1,
                            return_train_score=True,
                            verbose=1)

        print(f"Optimizing {target}...")
        gs_cv.fit(x_train, y_train_target)
        print(f"Migliori parametri per {target}:{gs_cv.best_params_}")
        
        best_model = gs_cv.best_estimator_
        best_params = gs_cv.best_params_
        models[target] = best_model

        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred

        mee_value = mean_euclidean_error(y_test_target, y_pred)
        print(f"MEE per il set di validazione: {mee_value}")


        # Calcola le metriche di errore usando i target originali
        print(f"Metriche per {target}:")
        print(f"Train Score del modello ottimale: {gs_cv.cv_results_['mean_train_score'][gs_cv.best_index_]}")
        print(f"Train standard deviation del modello ottimale: {gs_cv.cv_results_['std_train_score'][gs_cv.best_index_]}")

        print(f"Validation Score del modello ottimale: {gs_cv.cv_results_['mean_test_score'][gs_cv.best_index_]}")
        print(f"Validation standard deviation del modello ottimale: {gs_cv.cv_results_['std_test_score'][gs_cv.best_index_]}")

        #print(f"MAE: {mean_absolute_error(y_test_target, y_pred)}")
        print(f"MSE per il set di validazione: {mean_squared_error(y_test_target, y_pred)}")

        pred_test = best_model.predict(x_ts)


#ensamble con bagging
def ensamble_SVM2():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2=DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    x_tn, x_ts = p2.normalize_data(x_train_n, x_ts)
    models = {}
    predictions = {}

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        base_model = SVR()
        bagging_model = BaggingRegressor(base_model, n_estimators=50, random_state=42)
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        gs_cv = GridSearchCV(
            estimator=bagging_model,
            param_grid={
                'estimator__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                'estimator__C': [0.1, 1, 5, 10],
                'estimator__gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 1],
                'estimator__epsilon': [0.001, 0.01, 0.1, 0.5, 1]
            },
            cv=kfold,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            return_train_score=True,
            verbose=1
        )

        print(f"Optimizing {target}...")
        gs_cv.fit(x_train, y_train_target)
        print(f"Migliori parametri per {target}: {gs_cv.best_params_}")
        
        best_model = gs_cv.best_estimator_
        models[target] = best_model

        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred

        # Calcola le metriche di errore usando i target originali
        print(f"Metriche per {target}:")
        print(f"Train Score del modello ottimale: {gs_cv.cv_results_['mean_train_score'][gs_cv.best_index_]}")
        print(f"Train standard deviation del modello ottimale: {gs_cv.cv_results_['std_train_score'][gs_cv.best_index_]}")

        print(f"Validation Score del modello ottimale: {gs_cv.cv_results_['mean_test_score'][gs_cv.best_index_]}")
        print(f"Validation standard deviation del modello ottimale: {gs_cv.cv_results_['std_test_score'][gs_cv.best_index_]}")

        #print(f"MAE: {mean_absolute_error(y_test_target, y_pred)}")
        print(f"MSE per il set di validazione: {mean_squared_error(y_test_target, y_pred)}")

        pred_test = best_model.predict(x_ts)


#implementazione halving
def halvingFunction():
  ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  processor = DatasetProcessor(ROOT_DIR)
  x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
  x_ts = processor.read_ts()

  # Creazione dello scaler
  p2=DatasetProcessor2()
  x_train_scaled, x_test_scaled = p2.normalize_data(x_train_n, x_test_n)

  x_tn, x_ts = p2.normalize_data(x_train_n, x_ts)
  
  # Dizionari per modelli e predizioni
  models = {}
  predictions = {}

  for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
    y_train_target = y_train[:, i]
    y_test_target = y_test[:, i]

    model = SVR()

    # Definizione di una param_grid più grande
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],  # Inclusione del kernel 'poly'
        'C': [0.01, 0.1, 1, 10, 100, 1000],  # Più valori per il parametro C
        'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],  # Aggiunti valori come 'scale' e 'auto'
        'epsilon': [0.01, 0.1, 0.5, 1],  # Inclusione di epsilon
        'degree': [2, 3, 4],  # Grado del polinomio (per kernel 'poly')
        'coef0': [0.0, 0.1, 0.5, 1.0]  # Termini di bias per kernel non lineari ('poly')
    }

    # Inizializza HalvingGridSearchCV
    halving_cv = HalvingGridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # MSE negativo per regressione
        cv=3,  # Cross-validation
        factor=2,  # Riduzione progressiva (ogni iterazione elimina metà dei candidati)
        random_state=42,
        n_jobs=-1  # Usa tutti i processori disponibili
    )

    # Fitting del modello
    halving_cv.fit(x_train_scaled, y_train_target)

    # Migliori parametri trovati
    print(f"\n>>> Risultati per {target} <<<")
    print(f"Migliori parametri: {halving_cv.best_params_}")
    print(f"Miglior score (MSE negativo): {halving_cv.best_score_}")

    # Predizioni sul test set
    y_pred = halving_cv.best_estimator_.predict(x_test_scaled)

    mse_test = mean_squared_error(y_test_target, y_pred)
    print(f"MSE sul test set: {mse_test}")

    # Monitoraggio dei risultati migliori
    print("\nMonitoraggio dettagliato (migliori modelli):")
    results = halving_cv.cv_results_



def random_grid_search_xgboost():
    np.random.seed(42)
    random_state = 42

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)
    _, x_ts = p2.normalize_data(x_train_n, x_ts)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train = poly.fit_transform(x_train)
    x_test = poly.transform(x_test)
    x_ts = poly.transform(x_ts)    


    

    base_model = XGBRegressor(objective='reg:squarederror', random_state=random_state)
    multioutput_model = MultiOutputRegressor(base_model)

    # Aggiustamento specifico per TARGET_z
    print("Optimizing TARGET_z with specific parameters...")
    param_distributions_z = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],  # Più profondità
        'n_estimators': [300, 400, 500],  # Più estimatori
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],  # Regolarizzazione L1
        'reg_lambda': [1, 1.5, 2]  # Regolarizzazione L2
    }
     # Cross-validation e ricerca randomizzata
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    rs_cv_z = RandomizedSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=param_distributions_z,
        n_iter=20,
        cv=kfold,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1
    )

    rs_cv_z.fit(x_train, y_train[:, 2])
    print(f"Migliori parametri per TARGET_z: {rs_cv_z.best_params_}")
    y_pred_z = rs_cv_z.best_estimator_.predict(x_test)

    # Sostituisci le predizioni di TARGET_z con il modello specifico
    y_pred[:, 2] = y_pred_z


    print("Ottimizzazione del modello multioutput con XGBoost...")
    rs_cv.fit(x_train, y_train)  # Usa i pesi
    print(f"Migliori parametri: {rs_cv.best_params_}")

    best_model = rs_cv.best_estimator_
    y_pred = best_model.predict(x_test)

    print("Metriche complessive:")
    print(f"MSE complessivo: {mean_squared_error(y_test, y_pred, multioutput='uniform_average')}")
    mse_per_target = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        print(f"MSE per {target}: {mse_per_target[i]}")

    final_predictions = best_model.predict(x_ts)
    processor.write_blind_results(final_predictions)
    print("Risultati salvati correttamente!")


"""

def nested_grid_search_kfold():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()
    
    x_train, x_test = processor.normalize_data(x_train_n, x_test_n)
    x_tn, x_ts = processor.normalize_data(x_train_n, x_ts)
    
    models = {}
    predictions = {}
    statistics = {}
    
    # Custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    #plt.style.use('seaborn')
    
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        
        param_grid = {
            'C': np.linspace(0.1, 10, num=10),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': np.linspace(0.01, 1, num=5)
        }
        
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        model = SVR()
        
        nested_scores = []
        best_params_list = []

        # Standardizzazione dei dati
        #scaler = StandardScaler()
        #x_train_scaled = scaler.fit_transform(x_train)  # Standardizza il training set
        #x_test_scaled = scaler.transform(x_test)       # Applica la stessa trasformazione al test set

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(x_train_scaled)):
            # Divisione in fold
            X_train_fold, X_val_fold = x_train_scaled[train_idx], x_train_scaled[val_idx]
            y_train_fold, y_val_fold = y_train_target[train_idx], y_train_target[val_idx]
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=mee_scorer,  # Using MEE scorer
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
            
            print(f"Optimizing {target} (Fold {fold_idx})...")
            grid_search.fit(X_train_fold, y_train_fold)
            
            best_params = grid_search.best_params_
            print(f"Best parameters for {target} (Fold {fold_idx}): {best_params}")
            best_params_list.append(best_params)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val_fold)
            fold_predictions.append(y_pred)
            
            val_score = mee_score(y_val_fold, y_pred)  # Using MEE score
            nested_scores.append(-val_score)  # Negative because GridSearchCV maximizes
            
            print(f"MEE: {val_score}")
        
        # Calculate statistics
        fold_predictions = np.array(fold_predictions)
        mean_predictions = np.mean(fold_predictions, axis=0)
        std_predictions = np.std(fold_predictions, axis=0)
        
        statistics[target] = {
            'mean_score': np.mean(nested_scores),
            'std_score': np.std(nested_scores),
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions
        }
        
        # Plot 1: Score Distribution
        plt.subplot(2, 2, 1)
        sns.boxplot(data=nested_scores)
        plt.title('Cross-validation Score Distribution')
        plt.ylabel('Negative MEE')
        
        # Plot 2: Predictions vs Actual
        final_params = best_params_list[np.argmax(nested_scores)]
        final_model = SVR(**final_params)
        final_model.fit(x_train_scaled, y_train_target)  # Utilizza tutti i dati per il training finale
        models[target] = final_model

        y_test_pred = final_model.predict(x_test_scaled)  # Predizioni sul test set
        predictions[target] = y_test_pred
        
        # Print summary statistics
        print(f"\nSummary Statistics for {target}:")
        print(f"Mean Cross-validation MEE: {-statistics[target]['mean_score']:.4f}")  # Convert back to positive
        print(f"Standard Deviation of CV MEE: {statistics[target]['std_score']:.4f}")
        print(f"Mean Prediction Standard Deviation: {np.mean(statistics[target]['std_predictions']):.4f}")
        
        # Final predictions for test set
        y_ts = final_model.predict(x_ts)
        
    return models, predictions, statistics

"""

def nested_grid_search_kfold():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()
    
    # Normalize data
    x_train, x_test = processor.normalize_data(x_train_n, x_test_n)
    x_tn, x_ts = processor.normalize_data(x_train_n, x_ts)
    
    models, predictions, statistics = {}, {}, {}
    
    # Custom scoring function
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        
        param_grid = {
        'C': [0.1, 1, 10, 50, 100],  # Valori chiave per regolare la complessità del modello
        'kernel': ['rbf', 'poly'],  # Limitato a kernel rbf e poly per maggiore flessibilità
        'gamma': [0.01, 0.1, 1],  # Gamma più semplice per controllare la complessità
        'epsilon': [0.01, 0.05, 0.1]  # Range ristretto per migliorare la precisione predittiva
        }
        
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        model = SVR()
        
        nested_scores, best_params_list, fold_predictions = [], [], []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analysis for {target}', fontsize=16)
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(x_train)):
            X_train_fold, X_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train_target[train_idx], y_train_target[val_idx]
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=mee_scorer,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
            
            print(f"Optimizing {target} (Fold {fold_idx})...")
            grid_search.fit(X_train_fold, y_train_fold)
            
            best_params = grid_search.best_params_
            print(f"Best parameters for {target} (Fold {fold_idx}): {best_params}")
            best_params_list.append(best_params)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val_fold)
            fold_predictions.append(y_pred)
            
            val_score = mean_euclidean_error(y_val_fold, y_pred)
            nested_scores.append(-val_score)
            
            print(f"MEE: {val_score}")
        
        # Statistics
        fold_predictions = np.array(fold_predictions)
        statistics[target] = {
            'mean_score': np.mean(nested_scores),
            'std_score': np.std(nested_scores),
            'mean_predictions': np.mean(fold_predictions, axis=0),
            'std_predictions': np.std(fold_predictions, axis=0)
        }
        
        # Visualization
        sns.boxplot(data=nested_scores, ax=axes[0, 0])
        axes[0, 0].set_title('Cross-validation Score Distribution')
        axes[0, 0].set_ylabel('Negative MEE')
        
        final_params = best_params_list[np.argmax(nested_scores)]
        final_model = SVR(**final_params)
        final_model.fit(x_train, y_train_target)
        y_test_pred = final_model.predict(x_test)
        
        axes[0, 1].scatter(y_test_target, y_test_pred, alpha=0.5)
        axes[0, 1].plot([y_test_target.min(), y_test_target.max()], 
                        [y_test_target.min(), y_test_target.max()], 'r--', lw=2)
        axes[0, 1].set_title('Predictions vs Actual')
        
        errors = np.abs(y_test_pred - y_test_target)
        sns.histplot(errors, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Absolute Prediction Error Distribution')
        
        axes[1, 1].hist(statistics[target]['std_predictions'], bins=30)
        axes[1, 1].set_title('Standard Deviation of Predictions')
        
        plt.tight_layout()
        plt.savefig(f'analysis_{target}.png')
        plt.close()
        
        models[target] = final_model
        predictions[target] = y_test_pred
        
        print(f"\nSummary Statistics for {target}:")
        print(f"Mean Cross-validation MEE: {-statistics[target]['mean_score']:.4f}")
        print(f"Standard Deviation of CV MEE: {statistics[target]['std_score']:.4f}")
        print(f"Mean Prediction Standard Deviation: {np.mean(statistics[target]['std_predictions']):.4f}")
        
        y_ts_pred = final_model.predict(x_ts)
        processor.write_blind_results(y_ts_pred)
    
    return models, predictions, statistics

def random_grid_search_xgboost2():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)
    _, x_ts = p2.normalize_data(x_train_n, x_ts)

    # Definizione della griglia dei parametri per XGBoost
    param_distributions = {
        'estimator__learning_rate': np.random.uniform(0.01, 0.3, 10),
        'estimator__max_depth': np.random.randint(3, 10, 10),
        'estimator__n_estimators': np.random.randint(50, 300, 10),
        'estimator__subsample': np.random.uniform(0.5, 1.0, 10),
        'estimator__colsample_bytree': np.random.uniform(0.5, 1.0, 10),
        'estimator__gamma': np.random.uniform(0, 5, 10),
        'estimator__reg_alpha': np.random.uniform(0, 1, 10),
        'estimator__reg_lambda': np.random.uniform(0, 1, 10),
    }

    # Modello base con XGBoost
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # MultiOutputRegressor per gestire più target
    multioutput_model = MultiOutputRegressor(base_model)

    # Cross-validation e ricerca randomizzata
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    rs_cv = RandomizedSearchCV(
        estimator=multioutput_model,
        param_distributions=param_distributions,
        n_iter=100,  # Numero di iterazioni della ricerca
        cv=kfold,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    print("Ottimizzazione del modello multioutput con XGBoost...")
    rs_cv.fit(x_train, y_train)
    print(f"Migliori parametri: {rs_cv.best_params_}")

    best_model = rs_cv.best_estimator_

    # Predizioni sul set di test
    y_pred = best_model.predict(x_test)
    print("Metriche complessive per il modello multioutput:")
    print(f"MSE complessivo: {mean_squared_error(y_test, y_pred, multioutput='uniform_average')}")

    # Calcolo delle metriche per ciascun target
    mse_per_target = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        print(f"MSE per {target}: {mse_per_target[i]}")

    # Predizione su x_ts
    final_predictions = best_model.predict(x_ts)

    # Scrivi i risultati sul file
    processor.write_blind_results(final_predictions)
    print("Risultati salvati correttamente!")


def main():
    #random_grid_search()
    #random_grid_search2()
    #random_grid_search3()
    #optimized_random_grid_search()
    #optimized_random_grid_search2()
    #random_grid_search_different_feature_scaling()
    #grid_search()
    #ensamble_SVM2()
    #nested_grid_search_kfold()
    random_grid_search_multioutput()
    #halvingFunction()
    #random_grid_search_xgboost()


if __name__ == "__main__":
    main()