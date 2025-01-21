from keras.layers import CategoryEncoding
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split, ParameterSampler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, make_scorer
from imblearn.over_sampling import RandomOverSampler
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

# Definizione della funzione MEE
def mean_euclidean_error(y_true, y_pred):
    # Calcola la distanza euclidea media
    mee = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    
    # Converte il tensor in float (necessario per sklearn)
    return float(mee.numpy())


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


def random_grid_search_multioutput():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2 = DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)
    _, x_ts = p2.normalize_data(x_train_n, x_ts)

    # Definizione della griglia dei parametri
    param_distributions = {
        'estimator__kernel': ['linear', 'rbf', 'poly'],
        'estimator__C': np.random.uniform(0.1, 10, 20),
        'estimator__gamma': np.random.uniform(0.01, 1, 20),
        'estimator__epsilon': np.random.uniform(0.01, 0.1, 10)
    }

    # Modello base
    base_model = SVR()

    # MultiOutputRegressor per gestire più target
    multioutput_model = MultiOutputRegressor(base_model)

    # Cross-validation e ricerca randomizzata
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    rs_cv = RandomizedSearchCV(
        estimator=multioutput_model,
        param_distributions=param_distributions,
        n_iter=30,  # Numero di iterazioni della ricerca
        cv=kfold,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    print("Ottimizzazione del modello multioutput...")
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

        
#qui non applico standardizzazione a target (funziona meglio)
def random_grid_search():
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
    final_predictions = []  # Inizializza una lista vuota prima del ciclo

    # Creazione dello scorer personalizzato (valori negativi per coerenza con minimizzazione della loss)
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)

    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]

        model = SVR()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        rs_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=100,  # interessante cambiare
            cv=kfold,
            scoring=mee_scorer,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        print(f"Optimizing {target}...")
        rs_cv.fit(x_train, y_train_target)
        print(f"Migliori parametri per {target}: {rs_cv.best_params_}")

        best_model = rs_cv.best_estimator_
        best_params = rs_cv.best_params_
        models[target] = best_model

        y_pred = best_model.predict(x_test)
        predictions[target] = y_pred

        
        mee_value = mean_euclidean_error(y_test_target, y_pred)
        print(f"MEE per il set di validazione: {mee_value}")

        # Calcola le metriche di errore usando i target originali
        print(f"Metriche per {target}:")
        print(f"Train Score del modello ottimale: {rs_cv.cv_results_['mean_train_score'][rs_cv.best_index_]}")
        print(f"Train standard deviation del modello ottimale: {rs_cv.cv_results_['std_train_score'][rs_cv.best_index_]}")

        print(f"Validation Score del modello ottimale: {rs_cv.cv_results_['mean_test_score'][rs_cv.best_index_]}")
        print(f"Validation standard deviation del modello ottimale: {rs_cv.cv_results_['std_test_score'][rs_cv.best_index_]}")

        #print(f"MAE: {mean_absolute_error(y_test_target, y_pred)}")
        print(f"MSE per il set di validazione: {mean_squared_error(y_test_target, y_pred)}")
        # Predizione su x_ts per il target corrente
        pred_test = best_model.predict(x_ts)
        final_predictions.append(pred_test)

    # Combina tutte le predizioni in un array (N, 3)
    final_predictions = np.array(final_predictions).T  # Trasposta per avere (N, 3)

    # Scrivi i risultati sul file
    processor.write_blind_results(final_predictions)
    print("Risultati salvati correttamente!")
    #pred_test = rs_cv.predict(x_ts)
    #print(pred_test)
    #processor = DatasetProcessor(ROOT_DIR)
    #processor.write_blind_results(pred_test)


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


#funzione per nested        
def nested_grid_search_kfold():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    x_train_n, y_train, x_test_n, y_test = processor.read_tr(split=True)
    x_ts = processor.read_ts()

    # Creazione dello scaler
    p2=DatasetProcessor2()
    x_train, x_test = p2.normalize_data(x_train_n, x_test_n)

    x_tn, x_ts = p2.normalize_data(x_train_n, x_ts)
    
    # Dizionari per modelli e predizioni
    models = {}
    predictions = {}
    
    for i, target in enumerate(['TARGET_x', 'TARGET_y', 'TARGET_z']):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        # Parametri della grid search
        param_grid = {
            'C': np.linspace(0.1, 10, num=10),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': np.linspace(0.01, 1, num=5),
            'epsilon': np.linspace(0.01, 0.1, num=10)  
        }

        # Cross-validation
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        model = SVR()

        nested_scores = []
        best_params_list = []

        # Standardizzazione dei dati
        #scaler = StandardScaler()
        #x_train_scaled = scaler.fit_transform(x_train)  # Standardizza il training set
        #x_test_scaled = scaler.transform(x_test)       # Applica la stessa trasformazione al test set

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(x_train)):
            # Divisione in fold
            X_train_fold, X_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train_target[train_idx], y_train_target[val_idx]

            # Grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )

            print(f"Optimizing {target} (Fold {fold_idx})...")
            grid_search.fit(X_train_fold, y_train_fold)
            
            # Migliori parametri per il fold
            best_params = grid_search.best_params_
            print(f"Migliori parametri per {target} (Fold {fold_idx}): {best_params}")
            best_params_list.append(best_params)

            # Predizioni e metriche
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val_fold)
            val_score = mean_squared_error(y_val_fold, y_pred)
            nested_scores.append(-val_score)

            print(f"MSE: {val_score}")

        # Selezione del modello finale
        final_params = best_params_list[np.argmax(nested_scores)]
        print(f"\nParametri finali per {target}: {final_params}")

        final_model = SVR(**final_params)
        final_model.fit(x_train, y_train_target)  # Utilizza tutti i dati per il training finale
        models[target] = final_model

        y_test_pred = final_model.predict(x_test)  # Predizioni sul test set
        predictions[target] = y_test_pred
        
        print(f"Predizioni su x_test per {target}: {y_test_pred}")
        print(f"MSE per il set di validazione: {mean_squared_error(y_test_target, y_test_pred)}")

        y_ts = final_model.predict(x_ts)


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


from sklearn.multioutput import MultiOutputRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import os

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
    grid_search()
    #ensamble_SVM2()
    #nested_grid_search_kfold()
    #random_grid_search_multioutput()
    #halvingFunction()
    #random_grid_search_xgboost()


if __name__ == "__main__":
    main()