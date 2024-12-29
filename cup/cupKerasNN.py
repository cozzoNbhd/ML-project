import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cup.cupUtilities import DatasetProcessor


# Funzione per creare il modello Keras
def create_model(units_1, units_2, dropout_1, dropout_2, learning_rate):
    model = Sequential()
    model.add(Dense(units=units_1, activation='relu', input_shape=(12,)))
    model.add(Dropout(rate=dropout_1))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dropout(rate=dropout_2))
    model.add(Dense(3, activation='linear'))  # Output per TARGET_x, TARGET_y, TARGET_z

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse', metrics=['mae'])
    return model

# Funzione per addestrare il modello e restituire l'oggetto History
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    if x_val is not None and y_val is not None:
        # Con validazione
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    else:
        # Senza validazione
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

# Funzione per eseguire la Grid Search
def model_selection(x_train, y_train):
    param_grid = {
        'units_1': [64, 128],
        'units_2': [32, 64],
        'dropout_1': [0.2, 0.3],
        'dropout_2': [0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'epochs': [50, 100],
        'batch_size': [16, 32],
    }

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    best_model = None
    best_loss = float('inf')
    best_params = {}

    for units_1 in param_grid['units_1']:
        for units_2 in param_grid['units_2']:
            for dropout_1 in param_grid['dropout_1']:
                for dropout_2 in param_grid['dropout_2']:
                    for learning_rate in param_grid['learning_rate']:
                        for epochs in param_grid['epochs']:
                            for batch_size in param_grid['batch_size']:
                                print(f"Training model with units_1={units_1}, units_2={units_2}, "
                                      f"dropout_1={dropout_1}, dropout_2={dropout_2}, "
                                      f"learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}")
                                model = create_model(units_1, units_2, dropout_1, dropout_2, learning_rate)
                                history = train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size)

                                # Ottieni la perdita di validazione dall'oggetto History
                                val_loss = history.history['val_loss'][-1]

                                if val_loss < best_loss:
                                    best_loss = val_loss
                                    best_model = model
                                    best_params = {
                                        'units_1': units_1,
                                        'units_2': units_2,
                                        'dropout_1': dropout_1,
                                        'dropout_2': dropout_2,
                                        'learning_rate': learning_rate,
                                        'epochs': epochs,
                                        'batch_size': batch_size,
                                    }

    print("\nBest Parameters Found:", best_params)
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    test_data = processor.read_ts()

    if ms:
        params = model_selection(x_train, y_train)
    else:
        params = dict(units_1=128, units_2=64, dropout_1=0.2, dropout_2=0.2,
                      learning_rate=0.001, epochs=100, batch_size=16)

        # Crea il modello e addestra
    model = create_model(params['units_1'], params['units_2'], params['dropout_1'],
                         params['dropout_2'], params['learning_rate'])
    history = train_model(model, x_train, y_train, None, None, params['epochs'], params['batch_size'])

    # Plot della curva di apprendimento
    plot_learning_curve(history, savefig=True, **params)

    # Predizioni sul dataset cieco
    x_test = processor.read_ts()
    predictions = model.predict(x_test)
    print("\nPredizioni sul blind test:")
    print(predictions)


if __name__ == "__main__":
    keras_nn(ms=False)



