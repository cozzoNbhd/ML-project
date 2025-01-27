import optuna
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from optuna import trial
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
            layers.append(nn.ReLU())
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

    scaler = StandardScaler()

    # Carica i dataset
    x_train, y_train, x_test, y_test = processor.read_tr(split=True)
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_blind_test = processor.read_ts()
    x_tr_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.fit_transform(x_test)
    blind_test_norm = scaler.fit_transform(x_blind_test)

    # Converti in tensori
    x_tensor = torch.from_numpy(x_tr_norm).float().to(device)
    y_tensor = torch.from_numpy(y_train).float().to(device)
    x2_tensor = torch.from_numpy(x_train2).float().to(device)
    y2_tensor = torch.from_numpy(y_train2).float().to(device)
    x_val_tens = torch.from_numpy(x_val).float().to(device)
    y_val_tens = torch.from_numpy(y_val).float().to(device)
    x_tens = torch.from_numpy(blind_test_norm).float().to(device)
    x_test_tens = torch.from_numpy(x_test_norm).float().to(device)
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


def fit(model, optimizer, train_loader, val_loader, epochs=100, loss_fn=mean_euclidean_error):
    tr_losses, val_losses = [], []

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
            val_losses.append(np.mean(val_epoch_losses))

    return tr_losses, val_losses



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

def objective(trial, study):
    # Parametri da ottimizzare
    num_layers = trial.suggest_int("num_layers", 2, 5)
    num_units = trial.suggest_int("num_units", 20, 120)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    eta = trial.suggest_float("eta", 1e-3, 1e-1, log=True)
    lmb = trial.suggest_float("lmb", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [20, 30, 40, 50, 60])
    epochs = 100

    train_data, _, _, _, _, _, _, _ = set_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    euclidean_errors = []
    all_training_losses = []
    all_validation_losses = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        train_subset = Subset(train_data, train_idx)
        valid_subset = Subset(train_data, valid_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        model = NN(num_layers=num_layers, num_units=num_units, dropout_rate=dropout_rate).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=lmb)

        # Addestra il modello e raccogli le perdite
        tr_losses, val_losses= fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,  # Per cross-validation
            epochs=epochs,
            loss_fn=mean_euclidean_error
        )

        euclidean_errors.append(min(val_losses))
        all_training_losses.append(tr_losses)
        all_validation_losses.append(val_losses)

    # Salva le perdite nei `user_attrs` del trial corrente
    trial.set_user_attr("training_losses", all_training_losses)
    trial.set_user_attr("validation_losses", all_validation_losses)

    # Restituisce il negativo della media degli errori euclidei
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


def pytorch_nn(study=None):
    print("PyTorch training started...\n")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)

    # Inizializza i parametri
    params = None

    if study:
        # Estrai i migliori 5 trial
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        start_time = time.time()

        print("\nTop 10 Best Trials:")
        validation_losses_all = []
        for i, trial in enumerate(best_trials):
            training_losses = trial.user_attrs.get("training_losses", [])
            validation_losses = trial.user_attrs.get("validation_losses", [])
            
            # Controlla se ci sono dati
            if training_losses and validation_losses:
                train_means = [np.mean(fold) for fold in training_losses]
                train_stds = [np.std(fold) for fold in training_losses]
                val_means = [np.mean(fold) for fold in validation_losses]
                val_stds = [np.std(fold) for fold in validation_losses]

                validation_losses_all.extend(val_means)

                print(f"\nTrial {i + 1}:")
                print(f"  Params: {trial.params}")
                print(f"  Training Loss - Mean: {np.mean(train_means):.4f}, Std: {np.mean(train_stds):.4f}")
                print(f"  Validation Loss - Mean: {np.mean(val_means):.4f}, Std: {np.mean(val_stds):.4f}")
            else:
                print(f"\nTrial {i + 1}:")
                print(f"  Params: {trial.params}")
                print("  Training/Validation Loss data is missing.")
        if validation_losses_all:
            avg_validation_loss = np.mean(validation_losses_all)
            print(f"\nAverage of the Best 10 Trials' Validation Losses: {avg_validation_loss:.4f}")
        # Recupera i parametri del miglior trial
        best_trial = study.best_trial
        params = best_trial.params  # Assegna i parametri del miglior trial
        print("\nBest Trial:")
        print(f"  Params: {best_trial.params}")
        print(f"  Value (Objective): {best_trial.value:.4f}")

    else:
        print("No study provided. Using default parameters.")
        # Usa parametri di default se lo `study` non è fornito
        params = {
            "num_layers": 2,
            "num_units": 96,
            "dropout_rate": 0.0237811,
            "eta": 0.002035,
            "lmb": 0.0000109414,
            "batch_size": 20,
            "epochs": 200,
        }

    # Controllo finale per assicurarsi che i parametri siano definiti
    if not params:
        raise ValueError("Parameters could not be retrieved. Please check the study or provide default parameters.")

    # Dati
    train_data, _, _, _, _, x_blind_test, x_test, y_test = set_data()
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=params["batch_size"], shuffle=False)

    # Modello
    model = NN(
        num_layers=params["num_layers"],
        num_units=params["num_units"],
        dropout_rate=params["dropout_rate"],
    ).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["lmb"])

    training_losses = []
    test_losses = []
    epochs = 200

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Addestramento per un'epoca
        tr_loss = fit2(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            epochs=1,  # Una sola epoca per ogni chiamata
            loss_fn=mean_euclidean_error
        )

        # Salva il valore della perdita di training per l'epoca
        training_losses.append(tr_loss[-1])

        # Valutazione sul set di test
        model.eval()
        with torch.no_grad():
            epoch_test_losses = []
            for x_batch, y_batch in test_loader:
                y_pred = model(x_batch)
                test_loss = mean_euclidean_error(y_batch, y_pred).item()
                epoch_test_losses.append(test_loss)

            # Calcola la media delle perdite di test per l'epoca
            test_loss_mean = np.mean(epoch_test_losses)
            test_losses.append(test_loss_mean)

        # Stampa i risultati dell'epoca corrente
        print(f"Training Loss for Epoch {epoch + 1}: {training_losses[-1]:.4f}")
        print(f"Test Loss for Epoch {epoch + 1}: {test_losses[-1]:.4f}")

    print("\nPerforming predictions on blind test data...")
    y_pred, iloss = predict(model, x_blind_test)

    # Salvataggio dei risultati
    processor.write_blind_results(y_pred)
    print("Predictions on blind test data saved.")

    # Grafico delle curve di apprendimento
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss", color="blue", linestyle="-")
    plt.plot(test_losses, label="Test Loss", color="red", linestyle="--")
    plt.title("Training and Test Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("final_learning_curves.png")
    plt.show()

    # Risultati finali
    print("\nFinal Losses:")
    print(f"Training Loss: {training_losses[-1]:.4f}")
    print(f"Test Loss: {test_losses[-1]:.4f}")
    # Salva i risultati
    processor.write_blind_results(y_pred)
    print("Results saved.")

    """    # Plot delle curve di apprendimento
        plt.figure(figsize=(10, 6))
        plt.plot(tr_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.plot(test_losses, label="Internal Test Loss")
        plt.legend()
        plt.grid()
        plt.title("Learning Curve")
        plt.show()
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Crea lo study
    study = optuna.create_study(direction="maximize")

    # Ottimizza con lo study passato come extra
    study.optimize(lambda trial: objective(trial, study), n_trials=10)

    # Passa lo study a pytorch_nn
    pytorch_nn(study=study)