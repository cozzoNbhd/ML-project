import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from tensorflow.python.keras.regularizers import l2

from cupUtilities import DatasetProcessor
import tensorflow.keras.backend as K
from scikeras.wrappers import KerasClassifier, KerasRegressor


# Definizione della perdita Mean Euclidean Error
def mean_euclidean_error(y_true, y_pred):
    return K.sqrt(K.mean(K.sum(K.square(y_true - y_pred), axis=-1)))

# Funzione per creare il modello
def create_model(eta=0.003, alpha=0.4, lmb=0.0005, input_dim=12, units=128, num_layers=3,
                 kernel_initializer='glorot_normal', dropout=0.2, optimizer_type='SGD'):

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i in range(num_layers):
        model.add(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lmb),
                        kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        units //= 2  # Riduzione progressiva delle unità

    model.add(Dense(3, activation='linear', kernel_initializer=kernel_initializer))

    # Configurazione dell'ottimizzatore
    if optimizer_type == 'SGD':
        optimizer = SGD(learning_rate=eta, momentum=alpha)
    elif optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=eta)
    else:
        raise ValueError("Ottimizzatore non supportato")

    # Compilazione del modello
    model.compile(optimizer=optimizer, loss=mean_euclidean_error, metrics=['mse'])

    return model

# Funzione di selezione del modello
def model_selection(x, y, epochs=200):

    # Fissare il seed per la riproducibilità
    seed = 27
    np.random.seed(seed)

    # Scikeras wrapper
    model = KerasRegressor(model=create_model, verbose=0, epochs=epochs)

    """
    # Definizione della griglia di ricerca
    eta = np.arange(start=0.003, stop=0.01, step=0.001)
    eta = [float(round(i, 4)) for i in list(eta)]

    alpha = np.arange(start=0.4, stop=1, step=0.1)
    alpha = [float(round(i, 1)) for i in list(alpha)]

    lmb = np.arange(start=0.0005, stop=0.001, step=0.0001)
    lmb = [float(round(i, 4)) for i in list(lmb)]
    """

    eta = [0.003, 0.005, 0.007, 0.01]
    alpha = [0.4, 0.6, 0.8]
    lmb = [0.0005, 0.0007, 0.001]

    batch_size = [32, 64]
    num_layers = [2, 3]  # Numero di layer da provare
    units = [64, 128]  # Unita iniziali del primo layer
    dropout = [0.1, 0.2, 0.3]  # Valori di dropout da provare
    optimizer_type = ['SGD', 'Adam']

    param_grid = {
        'model__eta': eta,
        'model__alpha': alpha,
        'model__lmb': lmb,
        'model__input_dim': [x.shape[1]],
        'model__dropout': dropout,
        'model__num_layers': num_layers,
        'model__units': units,
        'model__optimizer_type': optimizer_type,
        'batch_size': batch_size
    }

    # Avvio della Grid Search
    start_time = time.time()
    print("Starting Grid Search...\n")


    #grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10,
    #                    return_train_score=True, scoring='neg_mean_squared_error', verbose=1, error_score='raise')


    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,  # Nota: usa "param_distributions" invece di "param_grid"
        n_iter=100,  # Numero di combinazioni casuali da testare
        n_jobs=-1,
        cv=10,
        return_train_score=True,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    grid_result = grid.fit(x, y)

    print("\nEnded Grid Search. ({:.4f} seconds)\n".format(time.time() - start_time))

    # Report dei risultati
    means_train = abs(grid_result.cv_results_['mean_train_score'])
    means_test = abs(grid_result.cv_results_['mean_test_score'])
    times_train = grid_result.cv_results_['mean_fit_time']
    times_test = grid_result.cv_results_['mean_score_time']
    params = grid_result.cv_results_['params']

    for m_ts, t_ts, m_tr, t_tr, p in sorted(zip(means_test, times_test, means_train, times_train, params)):
        print("{} \t TR {:.4f} (in {:.4f}s) \t TS {:.4f} (in {:.4f}s)".format(p, m_tr, t_tr, m_ts, t_ts))

    print("\nBest: {:.4f} using {}\n".format(abs(grid.best_score_), grid_result.best_params_))

    # Aggiungi il numero di epoche ai migliori parametri
    best_params = grid_result.best_params_
    best_params['epochs'] = epochs

    return best_params

# Funzione per la curva di apprendimento
def plot_learning_curve(history, start_epoch=1, savefig=False, **kwargs):
    lgd = ['Loss TR']
    plt.plot(range(start_epoch, kwargs['epochs'] + 1), history.history['loss'][start_epoch - 1:])
    if "val_loss" in history.history:
        plt.plot(range(start_epoch, kwargs['epochs'] + 1), history.history['val_loss'][start_epoch - 1:])
        lgd.append('Loss VL')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'Keras Learning Curve \n {kwargs}')
    plt.legend(lgd)

    if savefig:
        plt.savefig("keras_learning_curve.png")
    plt.show()

def keras_nn(ms=False):

    # Percorso radice del progetto (due livelli sopra il file corrente)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processor = DatasetProcessor(ROOT_DIR)

    # Carica il dataset di training con split
    x_train, y_train, x_test, y_test = processor.read_tr(split=True)

    test_data = processor.read_ts()

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    if ms:
        params = model_selection(x_train, y_train)
    else:
        params = dict(model__optimizer_type="SGD", model__num_layers=2, model__units=128, batch_size=64, model__alpha=0.9, model__dropout=0.1, model__eta=0.01, model__input_dim=12, model__lmb=0.0005, epochs=200)

    # Creazione del modello con i migliori iperparametri
    model = create_model(
        optimizer_type=params['model__optimizer_type'],
        num_layers=params['model__num_layers'],
        alpha=params['model__alpha'],
        eta=params['model__eta'],
        lmb=params['model__lmb'],
        input_dim=x_train.shape[1],
        units=params['model__units'],
        dropout=params['model__dropout'],
    )

    # Training del modello
    print("Inizio del training del modello...")
    history = model.fit(
        x_test,
        y_test,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=1
    )

    # Plot della curva di apprendimento
    plot_learning_curve(history, savefig=True, **params)

    # Predizioni sul dataset di test (blind test)
    print("Predizioni sul dataset di test...")
    predictions = model.predict(test_data)

    # Stampa delle predizioni
    print("Predizioni sul blind test:")
    print(predictions)

    # Salvataggio delle predizioni in un file CSV (opzionale)
    import pandas as pd
    predictions_df = pd.DataFrame(predictions, columns=['Output1', 'Output2', 'Output3'])
    predictions_df.to_csv("blind_test_predictions.csv", index=False)
    print("Predizioni salvate in 'blind_test_predictions.csv'.")


if __name__ == "__main__":
    keras_nn(ms=False)



