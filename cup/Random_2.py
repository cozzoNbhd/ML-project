import optuna
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from cupUtilities import DatasetProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rete Neurale Parametrizzata
class NN(nn.Module):
    def __init__(self, input_size=12, num_layers=2, num_units=40, dropout_rate=0.0):
        super(NN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, num_units))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_units, num_units))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_units, 3))  # 3 è il numero di output
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def set_data():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)

    # Carica i dataset
    x_train, y_train, x_test, y_test = processor.read_tr(split=True)
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_blind_test = processor.read_ts()

    # Converti in tensori
    x_tensor = torch.from_numpy(x_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).float().to(device)
    x2_tensor = torch.from_numpy(x_train2).float().to(device)
    y2_tensor = torch.from_numpy(y_train2).float().to(device)
    x_val_tens = torch.from_numpy(x_val).float().to(device)
    y_val_tens = torch.from_numpy(y_val).float().to(device)
    x_tens = torch.from_numpy(x_blind_test).float().to(device)
    x_test_tens = torch.from_numpy(x_test).float().to(device)
    y_test_tens = torch.from_numpy(y_test).float().to(device)


    train_data = TensorDataset(x_tensor, y_tensor)
    train_data2 = TensorDataset(x2_tensor,y2_tensor)
    val_data = TensorDataset(x_val_tens, y_val_tens)

    return train_data, train_data2, val_data, x_tensor, y_tensor, x_tens, x_test_tens, y_test_tens


def mean_euclidean_error(y_true, y_pred):
    return torch.mean(F.pairwise_distance(y_true, y_pred, p=2))

def model_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y, y_hat)
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


def fit(model, optimizer, train_loader, val_loader, test_loader, epochs=100, loss_fn=mean_euclidean_error):
    tr_losses, val_losses, test_losses = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        tr_losses.append(np.mean(epoch_losses))

        # Calcolo delle perdite di validazione
        model.eval()
        with torch.no_grad():
            val_epoch_losses = [loss_fn(model(x_val), y_val).item() for x_val, y_val in val_loader]
            test_epoch_losses = [loss_fn(model(x_test), y_test).item() for x_test, y_test in test_loader]
            val_losses.append(np.mean(val_epoch_losses))
            test_losses.append(np.mean(test_epoch_losses))

    return tr_losses, val_losses, test_losses



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

def objective(trial):
    # Parametri da ottimizzare (come prima)
    num_layers = trial.suggest_int("num_layers", 2, 5)     
    num_units = trial.suggest_int("num_units", 20, 100)    
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    eta = trial.suggest_float("eta", 1e-4, 1e-2, log=True)
    lmb = trial.suggest_float("lmb", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [10, 20, 30, 40])
    epochs = 100

    train_data, _, _, _, _, _, _, _ = set_data()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    euclidean_errors = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        # Creiamo i subset per questo fold
        train_subset = Subset(train_data, train_idx)
        valid_subset = Subset(train_data, valid_idx)
        
        # Creiamo i data loader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
        
        # Per la cross-validation, possiamo usare il validation set anche come test set
        # dato che stiamo solo cercando di ottimizzare gli iperparametri
        test_loader = val_loader
        
        # Inizializziamo e addestriamo il modello
        model = NN(num_layers=num_layers, num_units=num_units, dropout_rate=dropout_rate).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=lmb)
        
        # Chiamiamo fit con tutti gli argomenti richiesti
        _, val_losses, _ = fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            loss_fn=mean_euclidean_error
        )
        
        # Prendiamo il minimo errore euclideo medio
        min_euclidean_error = min(val_losses)
        euclidean_errors.append(min_euclidean_error)
    
    # Restituiamo il negativo della media degli errori
    return -np.mean(euclidean_errors)

def predict(model, x_ts):
    # change our data into tensors to work with PyTorch

    _ ,_, _, _, _, _, x_int_test, y_int_test = set_data()

    # predict on internal test set
    y_ipred = model(x_int_test)
    iloss = mean_euclidean_error(y_int_test, y_ipred)

    # predict on blind test set
    y_pred = model(x_ts).to(device)

    # return predicted target on blind test set,
    # and losses on internal test set
    return y_pred.detach().cpu().numpy(), iloss.item()


def pytorch_nn(use_optuna=True):
    print("PyTorch training started...\n")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)

    if use_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        # Migliori parametri
        params = study.best_params
        print(f"Best parameters: {params}")
    else:
        params = {
            "num_layers": 2,
            "num_units": 96,
            "dropout_rate": 0.0237811,
            "eta": 0.002035,
            "lmb": 0.0000109414,
            "batch_size": 20,
            "epochs": 200,
        }

    # Dati
    _, train_data2, val_data, _, _, x_blind_test, x_test, y_test = set_data()
    train_loader = DataLoader(train_data2, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=params["batch_size"], shuffle=False)

    # Modello
    model = NN(
        num_layers=params["num_layers"],
        num_units=params["num_units"],
        dropout_rate=params["dropout_rate"],
    ).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["lmb"])

    # Addestramento
    tr_losses, val_losses, test_losses = fit(model, optimizer, train_loader, val_loader, test_loader, params.get("epochs", 100))

    # Previsione
    y_pred, iloss = predict(model, x_blind_test)

    # Risultati
    print(f"Internal Test Loss: {iloss}")
    print(f"Final Training Loss: {tr_losses[-1]}")
    print(f"Final Validation Loss: {val_losses[-1]}")

    # Salva i risultati
    processor.write_blind_results(y_pred)
    print("Results saved.")

    # Plot delle curve di apprendimento
    plt.figure(figsize=(10, 6))
    plt.plot(tr_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(test_losses, label="Internal Test Loss")
    plt.legend()
    plt.grid()
    plt.title("Learning Curve")
    plt.show()

if __name__ == "__main__":
    pytorch_nn()
