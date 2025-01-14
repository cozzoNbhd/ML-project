import os
import multiprocessing as mp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold, train_test_split, ParameterGrid 
from sklearn.metrics import confusion_matrix
from skorch import NeuralNetClassifier
from torch import Tensor

from cupUtilities import DatasetProcessor
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Definiamo la Rete Neurale

class NN(nn.Module):
    def __init__(self, input_size=12, num_layers=2, num_units=40, dropout_rate=0.0):
        super(NN, self).__init__()
        layers = []

        # Primo layer
        layers.append(nn.Linear(input_size, num_units))
        
        # Aggiungi il numero di layer successivi
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_units, num_units))
            layers.append(nn.ReLU())  # Funzione di attivazione
            layers.append(nn.Dropout(dropout_rate))  # Dropout per ogni layer

        # Layer finale
        layers.append(nn.Linear(num_units, 3))  # 3 è il numero di output (adatta se necessario)

        # Combina tutti i layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def set():

    # Percorso radice del progetto (due livelli sopra il file corrente)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)

    # Carica il dataset di training con split
    x_train, y_train = processor.read_tr(split=False)
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    x_test, y_test = processor.read_tr(split=False)

    # change our data into tensors to work with PyTorch
    x_tensor = torch.from_numpy(x_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).float().to(device)
    x_tensor2 = torch.from_numpy(x_train2).float().to(device)
    y_tensor2 = torch.from_numpy(y_train2).float().to(device)
    x_val_tens = torch.from_numpy(x_val).float().to(device)
    y_val_tens = torch.from_numpy(y_val).float().to(device)
    x_int_tens = torch.from_numpy(x_test).float().to(device)
    y_int_tens = torch.from_numpy(y_test).float().to(device)

    train_data = TensorDataset(x_tensor2.to(device), y_tensor2.to(device))
    val_data = TensorDataset(x_val_tens.to(device), y_val_tens.to(device))


    return train_data, val_data, x_tensor, y_tensor, x_int_tens, y_int_tens

def mean_euclidean_error(y_true, y_hat): 
    return torch.mean(F.pairwise_distance(y_true, y_hat, p=2))

def model_train_step(model, loss_fn, optimizer):

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y,y_hat)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    return train_step

def plot_learning_curve(tr_losses, val_losses, savefig=True, **kwargs):
    start_epoch = kwargs.get('start_epoch', 0)
    epochs = len(tr_losses)  # Usa la lunghezza effettiva delle perdite come limite superiore

    # Assicurati che x e y abbiano la stessa lunghezza
    x = range(start_epoch, epochs)

    # Plotta le curve di apprendimento
    plt.figure(figsize=(10, 6))
    plt.plot(x, tr_losses[start_epoch:], label='Training Loss')
    plt.plot(x, val_losses[start_epoch:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig('learning_curve.png')
    plt.show()


def fit(model, optimizer, batch_size, loss_fn = mean_euclidean_error, epochs=130):

        train_step = model_train_step(model, loss_fn, optimizer)
        losses, val_losses = [], []

        train_data, val_data, _, _, _, _ = set()

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

        # per ciascuna epoca
        for epoch in range(epochs):
            epoch_losses = []
            for x_batch, y_batch in train_loader:
                loss = train_step(x_batch, y_batch)
                epoch_losses.append(loss)

            losses.append(np.mean(epoch_losses))

            epoch_val_losses =  []

            with torch.no_grad():
                for x_val, y_val in val_loader:

                    model.eval()
                    y_hat = model(x_val)

                    val_loss = loss_fn(y_val, y_hat)
                    epoch_val_losses.append(val_loss.item())

                val_losses.append(np.mean(epoch_val_losses))

        return losses, val_losses

def fit2(model, optimizer, train_loader, epochs, loss_fn=mean_euclidean_error):
    tr_losses = []  # Per salvare le perdite di addestramento

    for epoch in range(epochs):
        model.train()
        epoch_losses = []  # Per accumulare le perdite di ogni batch

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Resetta i gradienti
            y_pred = model(x_batch)  # Predizione
            loss = loss_fn(y_batch, y_pred)  # Calcolo della perdita
            loss.backward()  # Backpropagation
            optimizer.step()  # Aggiornamento dei pesi
            epoch_losses.append(loss.item())

        # Calcola la perdita media per l'epoca e aggiungila alla lista
        tr_losses.append(np.mean(epoch_losses))

    return tr_losses

def model_selection(x, y, model_class, loss_fn = mean_euclidean_error, epochs=120, n_splits = 7):
    best_loss = float("inf")
    best_params = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state =42)
    param_grid = {
        "batch_size": [10, 20, 30],
        "eta": [0.001, 0.002, 0.01],
        "dropout_rate": [0.0, 0.1],
        "lmb": [0.0005, 0.0007, 0.001],
        "num_layers": [2, 3],  # Esempio di grid per il numero di layer
        "num_units": [30, 40, 50]  # Esempio di grid per il numero di unità per layer
    }
    grid = ParameterGrid(param_grid)

    for param_dict in grid:
        avg_val_loss = 0

        for train_idx, val_idx in kf.split(x):
            x_train, x_val = x[train_idx].to(device), x[val_idx].to(device)
            y_train, y_val = y[train_idx].to(device), y[val_idx].to(device)

            model = model_class(num_layers=param_dict["num_layers"],num_units=param_dict["num_units"],dropout_rate=param_dict["dropout_rate"]).to(device)
            model.apply(init_weights)
            optimizer = optim.Adam(model.parameters(), lr=param_dict["eta"], weight_decay=param_dict["lmb"])

            for epoch in range(epochs):
                model.train()
                for i in range(0, len(x_train), param_dict["batch_size"]):
                    x_batch = x_train[i:i + param_dict["batch_size"]]
                    y_batch = y_train[i:i + param_dict["batch_size"]]

                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
        
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = loss_fn(val_outputs, y_val)
                avg_val_loss += val_loss

        avg_val_loss /= n_splits

        print(f"Params: {param_dict}, Avg Val Loss: {avg_val_loss}")
        
        # Aggiorna i migliori parametri se il modello ha ottenuto una loss migliore
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = param_dict


    return best_params, best_loss

def predict(model, x_ts):
    # change our data into tensors to work with PyTorch
    x_ts = torch.from_numpy(x_ts).float().to(device)

    _, _, _, _, x_int_test, y_int_test = set()

    # predict on internal test set
    y_ipred = model(x_int_test)
    iloss = mean_euclidean_error(y_int_test, y_ipred)

    # predict on blind test set
    y_pred = model(x_ts).to(device)

    # return predicted target on blind test set,
    # and losses on internal test set
    return y_pred.detach().cpu().numpy(), iloss.item()


def pytorch_nn(ms=True):
    print("pytorch start\n")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    
    params = None
    best_loss = None
    
    # choose model selection or hand-given parameters
    if ms:
        _, _, x_tensor, y_tensor, _, _ = set()
        params, best_loss = model_selection(x_tensor, 
        y_tensor,
        model_class=NN)
    
    else:
        params = dict(eta=0.001, lmb=0.0005, epochs=200, batch_size=10, dropout_rate=0.0)

    print(f"Best parameters: {params}, Loss: {best_loss}")
    # create and fit the model
    model = NN(dropout_rate=params["dropout_rate"]).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=params['eta'], weight_decay=params['lmb'])

    tr_losses, val_losses = fit2(model=model, optimizer=optimizer,
                                batch_size=params['batch_size'])

    y_pred, ts_losses = predict(model=model, x_ts=processor.read_ts())
    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    print("\npytorch end")

    plot_learning_curve(tr_losses, val_losses, savefig=True, **params)

    # generate csv file for MLCUP
    processor.write_blind_results(y_pred)


if __name__ == '__main__':
    pytorch_nn()