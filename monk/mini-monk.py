import numpy
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.layers import Dense, BatchNormalization, CategoryEncoding, Flatten,Input
from tensorflow.keras.models import Model
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
import itertools
from monkUtilities import DatasetProcessor  # Importa la classe dal file monkUtilities.py




def train_neural_network(train_path, test_path):
    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

    num_tokens = int(numpy.max(X_train) + 1)  # Numero di categorie (es. 4: 0, 1, 2, 3)

    # Applicazione di One-Hot Encoding su X_train, X_val e X_test
    encoding_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
    X_train_encoded = encoding_layer(X_train).numpy()
    X_val_encoded = encoding_layer(X_val).numpy()
    X_test_encoded = encoding_layer(X_test).numpy()


    # Appiattisci i dati codificati per renderli bidimensionali
    X_train_encoded = X_train_encoded.reshape(X_train_encoded.shape[0], -1)
    X_val_encoded = X_val_encoded.reshape(X_val_encoded.shape[0], -1)
    X_test_encoded = X_test_encoded.reshape(X_test_encoded.shape[0], -1)

    # Creazione del modello
    model = Sequential([
        Dense(4, activation='relu', kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.L2(0.0001)),  # Primo strato nascosto
        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])

    # Compilazione del modello
    model.compile(
        optimizer=SGD(learning_rate=0.3, momentum = 0.6),  # Ottimizzatore SGD
        loss=MeanSquaredError(),         # Loss Binary Cross Entropy
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
        epochs=200,
        batch_size=25,
        validation_data=(X_val_encoded, y_val),
        #callbacks=[early_stopping]
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist

def main():

    datasets = [
        ('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        ('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]


    for train_path, test_path in datasets:

        model, hist = train_neural_network(train_path, test_path)

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

if __name__ == "__main__":
    main()



