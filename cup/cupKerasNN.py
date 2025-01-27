import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam, SGD
from sklearn.metrics import make_scorer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from tensorflow.python.keras.regularizers import l2
from scipy.stats import uniform, randint

from cupUtilities import DatasetProcessor
import tensorflow.keras.backend as K
from scikeras.wrappers import KerasClassifier, KerasRegressor


def mean_euclidean_error(y_true, y_pred):
    """Calculate Mean Euclidean Error between true and predicted values."""
    return K.sqrt(K.mean(K.sum(K.square(y_true - y_pred), axis=-1)))

def make_mee_scorer():
    """Create a scorer function for scikit-learn that uses MEE."""
    def mee_sklearn(y_true, y_pred):
        return -float(K.get_value(mean_euclidean_error(y_true, y_pred)))
    return make_scorer(mee_sklearn)


def create_model(eta=0.003, alpha=0.4, lmb=0.0005, input_dim=12, units=128, num_layers=3,
                 kernel_initializer='glorot_normal', dropout=0.2, optimizer_type='SGD'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i in range(num_layers):
        model.add(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lmb),
                        kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(3, activation='linear', kernel_initializer=kernel_initializer))

    if optimizer_type == 'SGD':
        optimizer = SGD(learning_rate=eta, momentum=alpha)
    elif optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=eta)
    else:
        raise ValueError("Ottimizzatore non supportato")

    model.compile(optimizer=optimizer, loss=mean_euclidean_error)
    return model


def plot_all_losses(history, test_losses, title='Learning Curves'):
    """
    Plots training, validation, and test losses.

    Parameters:
    - history: Training history containing 'loss' and 'val_loss'.
    - test_losses: List of test losses recorded at each epoch.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss', color='blue')

    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green')

    # Plot test loss as a curve
    if isinstance(test_losses, list):
        plt.plot(test_losses, label='Internal Test Loss', color='red', linestyle='--')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.show()


def model_selection(x, y, epochs=200):

    seed = 27
    np.random.seed(seed)

    # Split the data into train, validation, and internal test sets
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=seed)

    model = KerasRegressor(model=create_model, verbose=0, epochs=epochs)

    param_grid = {
        # Learning rate (eta) as a continuous range between 0.001 and 0.01
        'model__eta': uniform(loc=0.001, scale=0.009),
        
        # Momentum (alpha) as a continuous range between 0.3 and 0.9
        'model__alpha': uniform(loc=0.3, scale=0.6),
        
        # L2 regularization (lmb) with a log-uniform distribution to sample values closer to zero
        'model__lmb': uniform(loc=0.0001, scale=0.0009),
        
        # Dropout rate as a continuous range
        'model__dropout': uniform(loc=0.1, scale=0.2),
        
        # Number of units in layers (integer sampling)
        'model__units': randint(32, 256),
        
        # Input dimension (fixed, but included for completeness)
        'model__input_dim': [x.shape[1]],
        
        # Number of layers (discrete integer sampling)
        'model__num_layers': randint(2, 5),
        
        # Batch size with discrete options
        'batch_size': [30, 40, 50, 60],
        
        # Optimizer type with discrete options
        'model__optimizer_type': ['SGD', 'Adam']
    }

    start_time = time.time()
    print("Starting Random Search...\n")

    # Create MEE scorer
    mee_scorer = make_mee_scorer()

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        n_jobs=-1,
        cv=10,
        return_train_score=True,
        scoring=mee_scorer,  # Use custom MEE scorer
        verbose=1
    )

    grid_result = grid.fit(x_train, y_train)

    print("\nEnded Random Search. ({:.4f} seconds)\n".format(time.time() - start_time))

    # Get and display top 10 results
    results = pd.DataFrame(grid_result.cv_results_)
    top_10 = results.nlargest(10, 'mean_test_score')

    print("\nTop 10 Model Performances:")
    for idx, row in top_10.iterrows():
        params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
        print(f"\nModel {idx + 1}:")
        print(f"Mean Training Loss: {-row['mean_train_score']:.4f} ± {row['std_train_score']:.4f}")
        print(f"Mean Validation Loss: {-row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"Parameters: {params}")

    best_params = grid_result.best_params_
    best_params['epochs'] = epochs
    return best_params

def keras_nn(ms=False):

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)

    # Load and split data
    x_full, y_full, x_holdout, y_holdout = processor.read_tr(split=True)

    # Further split the training data into train, validation, and internal test
    x_temp, x_test, y_temp, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

    test_data = processor.read_ts()

    if ms:
        params = model_selection(x_full, y_full)
    else:
        params = dict(model__optimizer_type="SGD", model__num_layers=3, model__units=128,
                      batch_size=64, model__alpha=0.6, model__dropout=0.1, model__eta=0.005,
                      model__input_dim=12, model__lmb=0.0007, epochs=200)

    # Create and train model
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

    # Inizializza una lista per registrare le perdite sul set di test
    test_losses = []

    print("Starting model training...")
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch + 1}/{params['epochs']}")

        # Addestra il modello per un'epoca
        history = model.fit(
            x_full, y_full,
            batch_size=params['batch_size'],
            epochs=1,
            verbose=1,
        )

        # Calcola la perdita sul set di test
        test_loss = model.evaluate(x_holdout, y_holdout, verbose=0)
        test_losses.append(test_loss)

        print(f"Test Loss for Epoch {epoch + 1}: {test_loss:.4f}")

        # Plot tutte le perdite
    plot_all_losses(history, test_losses, 'Training Learning Curves')

    print("\nFinal Losses:")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    #print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Internal Test Loss: {test_losses[-1]:.4f}")

    # Make predictions on blind test
    print("\nMaking predictions on blind test set...")
    predictions = model.predict(test_data)

    predictions_df = pd.DataFrame(predictions, columns=['Output1', 'Output2', 'Output3'])
    predictions_df.to_csv("blind_test_predictions.csv", index=False)
    print("Predictions saved to 'blind_test_predictions.csv'")


if __name__ == "__main__":
    keras_nn(ms=False)