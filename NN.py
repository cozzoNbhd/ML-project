import numpy
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, CategoryEncoding, Flatten,Input
from tensorflow.keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder

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



def train_neural_network2(train_path, test_path):
    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    print(X_train)
    print(len(X_val))
    

    model = Sequential([
        Dense(256, activation='relu', input_shape=(118,4), kernel_regularizer='l1'),  # Primo strato
        #BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer='l1'),  # Secondo strato
        #BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer='l1'),  # Terzo strato
        #BatchNormalization(),
        Dense(1, activation='sigmoid')  # Strato di output per classificazione binaria
    ])

    """
    # Creazione del modello
    model = Sequential([
        Dense(256, activation='relu', input_shape=(4,), kernel_regularizer=regularizers.l1(0.01)),  # Primo strato nascosto
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),  # Secondo strato nascosto
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)),  # Secondo strato nascosto
        BatchNormalization(),
        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])
    """
    # Compilazione del modello
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Ottimizzatore SGD
        loss=BinaryCrossentropy(),         # Loss Binary Cross Entropy
        metrics=['accuracy']               # Metrica Accuracy
    )

    # Definizione del callback per early stopping
    

    # Addestramento del modello
    hist = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        #callbacks=[early_stopping]
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist


def train_neural_network4(train_path, test_path):
    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)

    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    
    # Numero massimo di categorie (num_tokens)
    num_tokens = numpy.max(X_train) + 1  # Numero di categorie (es. 4: 0, 1, 2, 3)

    # Creazione del modello Keras con One-Hot Encoding
    input_layer = Input(shape=(X_train.shape[1],))  # Input delle feature originali
    encoded_layer = CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")(input_layer)  # One-Hot Encoding
    flatten_layer = Flatten()(encoded_layer)
    model = Sequential([
        Dense(256, activation='relu', input_shape=flatten_layer, kernel_regularizer='l1'),  # Primo strato
        #BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer='l1'),  # Secondo strato
        #BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer='l1'),  # Terzo strato
        #BatchNormalization(),
        Dense(1, activation='sigmoid')  # Strato di output per classificazione binaria
    ])

    
    # Compilazione del modello
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Ottimizzatore SGD
        loss=BinaryCrossentropy(),         # Loss Binary Cross Entropy
        metrics=['accuracy']               # Metrica Accuracy
    )

    # Definizione del callback per early stopping
    

    # Addestramento del modello
    hist = model.fit(
        X_train_encoded, y_train_seq,
        epochs=200,
        batch_size=32,
        validation_data=(X_val_encoded, y_val_seq),
        #callbacks=[early_stopping]
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test_encoded, y_test_seq)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist

def train_neural_network3(train_path, test_path):

    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)
    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    
    # Numero massimo di categorie (num_tokens)
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


    # Controllo delle forme dopo il One-Hot Encoding
    print(f"X_train_encoded shape: {X_train_encoded.shape}")
    print(f"X_val_encoded shape: {X_val_encoded.shape}")
    print(f"X_test_encoded shape: {X_test_encoded.shape}")

    # Creazione del modello Keras
    input_layer = Input(shape=(X_train_encoded.shape[1],)) 

    #dense1 = Dense(256, activation='relu')(flatten_layer)  # Primo strato denso
    #a1=BatchNormalization()(dense1)
    #dense2 = Dense(128, activation='relu')(dense1)  # Secondo strato denso
    #a2=BatchNormalization()(dense2)
    dense3 = Dense(128, activation='relu')(input_layer)  # Terzo strato denso
    dense2 = Dense(32, activation='relu')(dense3)
    output_layer = Dense(1, activation='sigmoid')(dense2)  # Strato di output per classificazione binaria

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compilazione del modello
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Ottimizzatore Adam
        loss=BinaryCrossentropy(),           # Loss Binary Cross Entropy
        metrics=['accuracy']                 # Metrica Accuracy
    )

    # Addestramento del modello
    hist = model.fit(
        X_train_encoded, y_train,
        epochs=100,  # Ridotto a 50 per velocità
        batch_size=32,
        validation_data=(X_val_encoded, y_val),
    )

    # Valutazione del modello
    loss, accuracy = model.evaluate(X_test_encoded, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    return model, hist




def main():
    datasets = [
        #('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        #('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]
    for train_path, test_path in datasets:
        model, hist = train_neural_network3(train_path, test_path)
     # Visualizza la curva di apprendimento
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()   

if __name__ == "__main__":
    main()




"""
model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Primo strato nascosto
        BatchNormalization(),
        Dense(64, activation='relu'),  # Secondo strato nascosto
        BatchNormalization(),
        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])

    # Compilazione del modello
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),  # Ottimizzatore SGD
        loss=BinaryCrossentropy(),         # Loss Binary Cross Entropy
        metrics=['accuracy']               # Metrica Accuracy
    )
"""