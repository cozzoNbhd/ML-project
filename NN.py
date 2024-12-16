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
import itertools
from utilities import DatasetProcessor  # Importa la classe dal file utilities.py




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
        optimizer=Adam(learning_rate=0.011, momentum = 0.9),  # Ottimizzatore SGD
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

def train_neural_network3(train_path, test_path):

    # Caricamento e preprocessing dei dati
    processor = DatasetProcessor()
    df_train, df_test = processor.load_dataset(train_path, test_path)

    X_train_full, y_train_full, X_test, y_test = processor.preprocess_data(df_train, df_test)
    # Dividi i dati di training in train e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    #X_train=[1,2,3,4]
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

    # Definisci i range per gli iperparametri
    learning_rates = numpy.arange(0.001, 0.1, 0.01)  # E.g., da 0.001 a 0.1 con step di 0.01
    batch_sizes = [16, 32, 64]  # I valori discreti sono ancora utili per il batch size
    num_units = numpy.arange(64, 256, 64)  # E.g., 64, 128, 192

    # Lista per salvare i risultati
    results = []

    # Ciclo per tutte le combinazioni degli iperparametri
    for lr, batch_size, units in itertools.product(learning_rates, batch_sizes, num_units):
        print(f"Testing combination: LR={lr}, Batch={batch_size}, Units={units}")
        
        # Crea il modello
        model = Sequential([
            Dense(units, activation='relu', input_shape=(X_train_encoded.shape[1],)),
            Dense(units // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # Addestra il modello
        hist = model.fit(
            X_train_encoded, y_train,
            epochs=20,
            batch_size=batch_size,
            validation_data=(X_val_encoded, y_val),
            verbose=0
        )
        
        # Salva i risultati
        val_accuracy = max(hist.history['val_accuracy'])
        results.append((lr, batch_size, units, val_accuracy))

    # Ordina i risultati per val_accuracy
    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
    print("Best combination:", sorted_results[0])



    """

    # Creazione del modello Keras
     
    
    # Creazione del modello
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_encoded.shape[1],)),  # Primo strato nascosto
        Dense(32, activation='relu'),  # Secondo strato nascosto
        Dense(1, activation='sigmoid')  # Strato di output con sigmoid
    ])
   
    #ALTRO MODO PER SCRIVERE MODELLI 
    #input_layer = Input(shape=(X_train_encoded.shape[1],))
    #dense3 = Dense(128, activation='relu')(input_layer)  # Terzo strato denso
    #dense2 = Dense(32, activation='relu')(dense3)
    #output_layer = Dense(1, activation='sigmoid')(dense2)  # Strato di output per classificazione binaria
    #model = Model(inputs=input_layer, outputs=output_layer)

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

    return model, hist"""


def main():
    datasets = [
        ('./datasets/monk/monks-1.train', './datasets/monk/monks-1.test'),
        ('./datasets/monk/monks-2.train', './datasets/monk/monks-2.test'),
        ('./datasets/monk/monks-3.train', './datasets/monk/monks-3.test')
    ]
    for train_path, test_path in datasets:
        train_neural_network3(train_path, test_path)
        # Visualizza la curva di apprendimento
        plt.plot(hist.history['accuracy'], label='Train Accuracy')
        plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.plot(hist.history['loss'], label='Train loss')
        plt.plot(hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()   

if __name__ == "__main__":
    main()



