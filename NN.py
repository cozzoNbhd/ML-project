import numpy
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from utilities import DatasetProcessor  # Importa la classe dal file utilities.py

def train_neural_network(train_path, test_path):
    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    # Creazione del modello
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Primo strato nascosto
        Dense(32, activation='relu'),  # Secondo strato nascosto
        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])

    # Compilazione del modello
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),  # Ottimizzatore SGD
        loss=BinaryCrossentropy(),         # Loss Binary Cross Entropy
        metrics=['accuracy']               # Metrica Accuracy
    )

    # Definizione del callback per early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitoriamo la perdita sulla validazione
        patience=100,  # Numero di epoche da aspettare senza miglioramenti
        restore_best_weights=True  # Ripristina i pesi migliori al termine
    )

    # Addestramento del modello
    model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val),
        #callbacks=[early_stopping]
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model


def main():
    datasets = [
        ('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        ('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]
    for train_path, test_path in datasets:
        train_neural_network(train_path, test_path)


if __name__ == "__main__":
    main()