import os
import multiprocessing as mp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from skorch import NeuralNetClassifier

from cupUtilities import DatasetProcessor
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Definiamo la Rete Neurale
ms_result =[]

class NN(nn.Module):
    def __init__(self, input_size=12, n_units=40, n_output=3, dropout_rate=0.2):
        super(NN, self).__init__()
        self.ly_in = nn.Linear(input_size, n_units)
        
        self.ly1 = nn.Linear(n_units, n_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.ly2 = nn.Linear(n_units, n_units)
        self.dropout = nn.Dropout(dropout_rate)

        self.ly_out = nn.Linear(n_units, out_features=n_output)

    def forward(self, x):

        x = F.relu(self.ly_in(x))

        x = F.relu(self.ly1(x))
        x = F.relu(self.ly2(x))

        x = self.ly_out(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def set(batch_size):

    # Percorso radice del progetto (due livelli sopra il file corrente)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processor = DatasetProcessor(ROOT_DIR)

    # Carica il dataset di training con split
    x_train, y_train = processor.read_tr(split=False)
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    x_test, y_test = processor.read_tr(split=False)

    # change our data into tensors to work with PyTorch
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()

    x_tensor2 = torch.from_numpy(x_train2).float()
    y_tensor2 = torch.from_numpy(y_train2).float()

    x_val_tens = torch.from_numpy(x_val).float()
    y_val_tens = torch.from_numpy(y_val).float()

    x_int_tens = torch.from_numpy(x_test).float()
    y_int_tens = torch.from_numpy(y_test).float()

    train_data = TensorDataset(x_tensor2, y_tensor2)
    val_data = TensorDataset(x_val_tens, y_val_tens)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, x_tensor, y_tensor, x_int_tens, y_int_tens

def mean_euclidean_error(y_true, y_hat):
    return torch.mean(F.pairwise_distance(y_true, y_hat, p=2))

def model_train_step(model, loss_fn, optimizer):

    def train_step(x, y):

        model.train()
        y_hat = model(x)

        loss = loss_fn(y,y_hat)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    
    return train_step

def plot_learning_curve(losses, val_losses, start_epoch=1, savefig=False, **kwargs):
    plt.plot(range(start_epoch, kwargs['epochs']), losses[start_epoch:])
    plt.plot(range(start_epoch, kwargs['epochs']), val_losses[start_epoch:])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['Loss TR', 'Loss VL'])
    plt.title(f'PyTorch Learning Curve \n {kwargs}')

    if savefig:
        save_figure("pytorchNN", **kwargs)

    plt.show()

def fit(model, optimizer, loss_fn = mean_euclidean_error, epochs=150, batch_size=40):

        train_step = model_train_step(model, loss_fn, optimizer)
        losses = []
        val_losses = []

        train_loader, val_loader = set(batch_size=batch_size)

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


def model_selection(loss_fn=mean_euclidean_error):
    
    x_tensor, y_tensor = set(batch_size=40)

    model = NeuralNetClassifier(
        NN,
        criterion=loss_fn,
        optimizer=optim.Adam,
        max_epochs=200,
        verbose=False
    )
    param_grid = {
        "batch_size": [10, 20, 30, 40],
        "eta": [0.001, 0.01, 0.05, 0.1],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3],
        "lmb": [0.0005, 0.001]

    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=8)
    grid_result = grid.fit(x_tensor, y_tensor)
    best_params = grid_result.best_params_

    print("Best: %f using %s" (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return best_params

def predict(model, x_ts):
    # change our data into tensors to work with PyTorch
    x_ts = torch.from_numpy(x_ts).float()

    x_int_test, y_int_test = set()

    # predict on internal test set
    y_ipred = model(x_int_test)
    iloss = mean_euclidean_error(y_int_test, y_ipred)

    # predict on blind test set
    y_pred = model(x_ts)

    # return predicted target on blind test set,
    # and losses on internal test set
    return y_pred.detach().numpy(), iloss.item()


def pytorch_nn(ms=True):
    print("pytorch start\n")

    # read training set
    #x_ts = DatasetProcessor.read_ts()
    # choose model selection or hand-given parameters
    if ms:
        params = model_selection()
    else:
        params = dict(eta=0.003, alpha=0.85, lmb=0.0002, epochs=80, batch_size=64)

    # create and fit the model
    model = NN()
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=params['eta'], weight_decay=params['lmb'])

    tr_losses, val_losses = fit(model=model, optimizer=optimizer,
                                batch_size=params['batch_size'], dropout_rate=params["dropout_rate"])

    y_pred, ts_losses = predict(model=model, x_ts=DatasetProcessor.read_ts())

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    print("\npytorch end")

    plot_learning_curve(tr_losses, val_losses, savefig=True, **params)

    # generate csv file for MLCUP
    DatasetProcessor.write_blind_results(y_pred)


if __name__ == '__main__':
    pytorch_nn()