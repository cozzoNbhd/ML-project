import os
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold

from cupUtilities import DatasetProcessor
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Definiamo la Rete Neurale

class NN(nn.Module):
    def __init__(self, input_size=12, n_units=40, n_output=3):
        super(NN, self).__init__()
        self.ly_in = nn.Linear(input_size, n_units)
        
        self.ly1 = nn.Linear(n_units, n_units)
        self.ly2 = nn.Linear(n_units, n_units)

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
        x_train, y_train = processor.read_tr(split=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0., random_state=42)
        
        # change our data into tensors to work with PyTorch
        x_tensor = torch.from_numpy(x_train).float()
        y_tensor = torch.from_numpy(y_train).float()

        x_val_tens = torch.from_numpy(x_val).float()
        y_val_tens = torch.from_numpy(y_val).float()

        train_data = TensorDataset(x_tensor, y_tensor)
        val_data = TensorDataset(x_val_tens, y_val_tens)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    def mean_euclidean_error(y_true, y_hat):
        return torch.mean(F.pairwise_distance(y_true, y_hat, p=2))
    