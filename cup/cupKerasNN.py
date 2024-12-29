import numpy
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.layers import Dense, CategoryEncoding
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from keras import regularizers
import itertools
from monk.monkUtilities import DatasetProcessor  # Importa la classe dal file monkUtilities.py




def train_neural_network(train_path, test_path):
    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    num_tokens = int(numpy.max(X_train) + 1)  # Numero di categorie (es. 4: 0, 1, 2, 3)

    # Applicazione di One-Hot Encoding su X_train, X_val e X_test
    encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
    X_train_encoded = encoding_layer(X_train).numpy()
    X_val_encoded = encoding_layer(X_val).numpy()
    X_test_encoded = encoding_layer(X_test).numpy()

    print(X_train_encoded.shape)

    # Appiattisci i dati codificati per renderli bidimensionali
    X_train_encoded = X_train_encoded.reshape(X_train_encoded.shape[0], -1)
    X_val_encoded = X_val_encoded.reshape(X_val_encoded.shape[0], -1)
    X_test_encoded = X_test_encoded.reshape(X_test_encoded.shape[0], -1)

    print(X_train_encoded.shape)
    # Creazione del modello
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_encoded.shape[1],),  kernel_regularizer=regularizers.L2(0.01)),  # Primo strato nascosto

        Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),  # Secondo strato nascosto

        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])

    # Compilazione del modello
    model.compile(
        optimizer=SGD(learning_rate=0.011, momentum = 0.9),  # Ottimizzatore SGD
        loss=BinaryCrossentropy(),         # Loss Binary Cross Entropy
        metrics=['accuracy'] ,              # Metrica Accuracy

    )

    # Definizione del callback per early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitoriamo la perdita sulla validazione
        patience=15,  # Numero di epoche da aspettare senza miglioramenti
        restore_best_weights=True  # Ripristina i pesi migliori al termine
    )

    # Addestramento del modello
    hist = model.fit(
        X_train_encoded, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val_encoded, y_val),
        callbacks=[early_stopping]
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist


def linear_decay_schedule(initial_lr, final_lr, total_epochs):
    """
    Restituisce una funzione che implementa il learning rate decay lineare.

    Args:
        initial_lr (float): Learning rate iniziale.
        final_lr (float): Learning rate finale.
        total_epochs (int): Numero totale di epoche.

    Returns:
        function: Funzione da usare nel callback LearningRateScheduler.
    """
    def scheduler(epoch, lr):
        new_lr = initial_lr - (epoch / total_epochs) * (initial_lr - final_lr)
        return max(new_lr, final_lr)
    return scheduler

def train_neural_network_with_grid_search(train_path, test_path, params):
    # Parametri di input
    learning_rates = params.get('learning_rates', [0.001])
    batch_sizes = params.get('batch_sizes', [32])
    num_units = params.get('num_units', [64])
    epochs = params.get('epochs', 50)
    patience = params.get('patience', 15)
    optimizer_choice = params.get('optimizer', 'adam')
    regularization = params.get('regularization', 0.01)
    final_lr = params.get('final_lr', 0.0001)  # Learning rate minimo finale

    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    # Encoding delle feature
    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train.shape[1]):
        num_tokens = int(np.max(X_train[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train[:, i]).numpy())
        encoded_val_list.append(encoding_layer(X_val[:, i]).numpy())
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)
    X_val_encoded = np.concatenate(encoded_val_list, axis=1)
    X_test_encoded = np.concatenate(encoded_test_list, axis=1)

    # Ricerca della miglior combinazione di iperparametri
    results = []
    for lr, batch_size, units in itertools.product(learning_rates, batch_sizes, num_units):
        if optimizer_choice.lower() == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_choice.lower() == 'sgd':
            optimizer = SGD(learning_rate=lr, momentum=0.9)

        model = Sequential([
            Dense(units, activation='relu', input_shape=(X_train_encoded.shape[1],),
                  kernel_regularizer=regularizers.L2(regularization)),
            Dense(units // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

        # Callback per learning rate decay lineare
        lr_scheduler = LearningRateScheduler(linear_decay_schedule(lr, final_lr, epochs))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        hist = model.fit(X_train_encoded, y_train, epochs=20, batch_size=batch_size,
                         validation_data=(X_val_encoded, y_val), verbose=0, callbacks=[lr_scheduler, early_stopping])
        val_accuracy = max(hist.history['val_accuracy'])
        results.append((lr, batch_size, units, val_accuracy))

    # Selezione dei migliori iperparametri
    best_lr, best_batch, best_units, _ = sorted(results, key=lambda x: x[3], reverse=True)[0]
    print(f"Migliori iperparametri: LR={best_lr}, Batch={best_batch}, Units={best_units}")

    # Addestramento finale
    optimizer = Adam(learning_rate=best_lr) if optimizer_choice.lower() == 'adam' else SGD(learning_rate=best_lr,
                                                                                           momentum=0.9)
    model = Sequential([
        Dense(best_units, activation='relu', input_shape=(X_train_encoded.shape[1],),
              kernel_regularizer=regularizers.L2(regularization)),
        Dense(best_units // 2, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    hist = model.fit(X_train_encoded, y_train, epochs=epochs, batch_size=best_batch,
                     validation_data=(X_val_encoded, y_val), callbacks=[early_stopping], verbose=1)


    # Valutazione finale
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist

def train_neural_network_with_kfold(train_path, test_path, params, k_folds=5):

    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)


    # Encoding delle feature
    encoded_train_list, encoded_val_list, encoded_test_list = [], [], []
    for i in range(X_train_full.shape[1]):
        num_tokens = int(np.max(X_train_full[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train_full[:, i]).numpy())


    X_train_encoded = np.concatenate(encoded_train_list, axis=1)

    # Parametri fissi
    lr = params.get('learning_rate', 0.01)
    batch_size = params.get('batch_size', 32)
    num_units = params.get('num_units', 64)
    epochs = params.get('epochs', 50)
    patience = params.get('patience', 15)
    optimizer_choice = params.get('optimizer', 'adam')
    regularization = params.get('regularization', 0.01)
    final_lr = params.get('final_lr', 0.0001)  # Learning rate minimo finale

    # Preparazione K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded)):
        print(f"Fold {fold + 1}/{k_folds}")
        X_train, X_val = X_train_encoded[train_idx], X_train_encoded[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Configurazione ottimizzatore
        optimizer = Adam(learning_rate=lr) if optimizer_choice.lower() == 'adam' else SGD(learning_rate=lr, momentum=0.9)

        # Creazione del modello
        model = Sequential([
            Dense(num_units, activation='relu', input_shape=(X_train.shape[1],),
                  kernel_regularizer=regularizers.L2(regularization)),
            Dense(num_units // 2, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

        # Callback per learning rate decay lineare
        lr_scheduler = LearningRateScheduler(linear_decay_schedule(lr, final_lr, epochs))

        # Callback per early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Addestramento del modello
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping], verbose=1)


        # Valutazione sul fold corrente
        val_accuracy = max(hist.history['val_accuracy'])
        fold_results.append(val_accuracy)
        print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy}")

    # Media dei risultati su tutti i folds
    mean_accuracy = np.mean(fold_results)
    print(f"Mean Validation Accuracy across {k_folds} folds: {mean_accuracy}")

    # Addestramento finale sul dataset completo
    print("Training final model on full training data...")
    optimizer = Adam(learning_rate=lr) if optimizer_choice.lower() == 'adam' else SGD(learning_rate=lr, momentum=0.9)
    model = Sequential([
        Dense(num_units, activation='relu', input_shape=(X_train_encoded.shape[1],),
              kernel_regularizer=regularizers.L2(regularization)),
        Dense(num_units // 2, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
        Dense(1, activation='sigmoid')
    ])

    # Callback per early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    hist = model.fit(X_train_encoded, y_train_full, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

    # Valutazione sul test set
    print("Evaluating on test set...")
    encoded_test_list = []
    for i in range(X_test.shape[1]):
        num_tokens = int(np.max(X_test[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_test_encoded = np.concatenate(encoded_test_list, axis=1)
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist

def train_neural_network_with_kfold_and_gridsearch(train_path, test_path, params, k_folds=5):
    # Parametri di input
    learning_rates = params.get('learning_rates', [0.001])
    batch_sizes = params.get('batch_sizes', [32])
    num_units = params.get('num_units', [64])
    epochs = params.get('epochs', 50)
    patience = params.get('patience', 15)
    optimizer_choice = params.get('optimizer', 'adam')
    regularization = params.get('regularization', 0.01)
    final_lr = params.get('final_lr', 0.0001)  # Learning rate minimo finale


    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)
    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Encoding delle feature
    encoded_train_list = []
    for i in range(X_train_full.shape[1]):
        num_tokens = int(np.max(X_train_full[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_train_list.append(encoding_layer(X_train_full[:, i]).numpy())

    X_train_encoded = np.concatenate(encoded_train_list, axis=1)

    # Preparazione K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Ricerca della miglior combinazione di iperparametri
    results = []

    for lr, batch_size, units in itertools.product(learning_rates, batch_sizes, num_units):

        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded)):
            print(f"Grid Search - LR: {lr}, Batch: {batch_size}, Units: {units}, Fold {fold + 1}/{k_folds}")

            X_train, X_val = X_train_encoded[train_idx], X_train_encoded[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            # Configurazione ottimizzatore
            optimizer = Adam(learning_rate=lr) if optimizer_choice.lower() == 'adam' else SGD(learning_rate=lr, momentum=0.9)

            # Creazione del modello
            model = Sequential([
                Dense(units, activation='relu', input_shape=(X_train.shape[1],),
                      kernel_regularizer=regularizers.L2(regularization)),
                Dense(units // 2, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

            # Callback per early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            # Addestramento del modello
            hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

            # Valutazione sul fold corrente
            val_accuracy = max(hist.history['val_accuracy'])
            fold_accuracies.append(val_accuracy)



        # Media delle accuratezze tra i fold
        mean_fold_accuracy = np.mean(fold_accuracies)
        results.append((lr, batch_size, units, mean_fold_accuracy))
        print(f"Mean Fold Accuracy for LR={lr}, Batch={batch_size}, Units={units}: {mean_fold_accuracy}")


    # Selezione dei migliori iperparametri
    best_lr, best_batch, best_units, _ = sorted(results, key=lambda x: x[3], reverse=True)[0]
    print(f"Migliori iperparametri: LR={best_lr}, Batch={best_batch}, Units={best_units}")

    # Addestramento finale con i migliori iperparametri
    optimizer = Adam(learning_rate=best_lr) if optimizer_choice.lower() == 'adam' else SGD(learning_rate=best_lr, momentum=0.9)

    model = Sequential([
        Dense(best_units, activation='relu', input_shape=(X_train_encoded.shape[1],),
              kernel_regularizer=regularizers.L2(regularization)),
        Dense(best_units // 2, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

    # Addestramento finale su tutto il set di addestramento
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    hist = model.fit(X_train_encoded, y_train_full, epochs=epochs, batch_size=best_batch,
                     validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Valutazione finale sul test set
    print("Evaluating on test set...")
    encoded_test_list = []
    for i in range(X_test.shape[1]):
        num_tokens = int(np.max(X_test[:, i]) + 1)
        encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        encoded_test_list.append(encoding_layer(X_test[:, i]).numpy())

    X_test_encoded = np.concatenate(encoded_test_list, axis=1)
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist


def main():

    datasets = [
        ('../datasets/monk/monks-1.train', '../datasets/monk/monks-1.test'),
        ('../datasets/monk/monks-2.train', '../datasets/monk/monks-2.test'),
        ('../datasets/monk/monks-3.train', '../datasets/monk/monks-3.test')
    ]

    # Configurazione dei parametri in caso di grid search
    params = {
        'learning_rates': [0.001, 0.01, 0.1],
        'batch_sizes': [16, 32, 64],
        'num_units': [16, 32, 64, 128],
        'epochs': 100,
        'patience': 20,
        'optimizer': 'adam',
        'regularization': 0.001,
        'final_lr': 0.0001
    }

    # Configurazione dei parametri in caso di k-fold normale (parametri fissi)
    params = {
        'learning_rates': 0.001,
        'batch_sizes': 16,
        'num_units': 32,
        'epochs': 100,
        'patience': 20,
        'optimizer': 'adam',
        'regularization': 0.001,
        'final_lr': 0.0001
    }

    for train_path, test_path in datasets:

        model, hist = train_neural_network_with_kfold(train_path, test_path, params)

        # Grafico Accuracy
        plt.figure()
        plt.plot(hist.history['accuracy'], label='Train Accuracy')
        plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Accuracy for {train_path}")
        plt.show()

        # Grafico Loss
        plt.figure()
        plt.plot(hist.history['loss'], label='Train Loss')
        plt.plot(hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Loss for {train_path}")
        plt.show()


    """ SOLO K-FOLD
    for train_path, test_path in datasets:
        model, hist = train_neural_network_with_grid_search(train_path, test_path, params)

        # Grafico Accuracy
        plt.figure()
        plt.plot(hist.history['accuracy'], label='Train Accuracy')
        plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Accuracy for {train_path}")
        plt.show()

        # Grafico Loss
        plt.figure()
        plt.plot(hist.history['loss'], label='Train Loss')
        plt.plot(hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Loss for {train_path}")
        plt.show()
    """

    """ K-FOLD E GRID SEARCH
    
    
    """
if __name__ == "__main__":
    main()



