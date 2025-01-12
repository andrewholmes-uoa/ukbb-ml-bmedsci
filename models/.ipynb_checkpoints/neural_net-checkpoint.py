import torch
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, BatchNorm1d, ReLU, Sigmoid
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import random
import numpy as np
import copy

activation_function_str_dict = {
    'ReLU': ReLU,
    'Sigmoid': Sigmoid,
}

def reset_weights(m):
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          layer.reset_parameters()

def process_X(X):
    return tensor(X.values).float()

def process_y(y):
    y = tensor(y.values).float()
    y = y.reshape((y.shape[0], 1))
    return y

class MLP(nn.Module):
    def __init__(self, input_layer_size, n_hidden_layers, hidden_layer_size, dropout, activation_function, optimizer_params, n_epochs, batch_size, n_jobs):
        super().__init__()

        if type(activation_function) == str:
            activation_function = activation_function_str_dict[activation_function]
            # (avoid passing function in params for Optuna - generates warnings)

        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_layer_size, hidden_layer_size))
            else:
                layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))    
            layers.append(BatchNorm1d(hidden_layer_size))
            layers.append(activation_function())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layer_size, 1))
        self.layers = nn.Sequential(*layers)

        self.optimizer_params = optimizer_params
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        
        self.float()

    def forward(self, x):
        return self.layers(x)


    def fit(self, X_train, y_train):
        self.apply(reset_weights)
        torch.set_num_threads(8)

        torch.manual_seed(2024)
        random.seed(2024)
        np.random.seed(2024)
        
        loss_function = BCEWithLogitsLoss()
        optimizer = AdamW(self.parameters(), **self.optimizer_params)

        train_dataset = TensorDataset(process_X(X_train), process_y(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs)

        for epoch in range(0, self.n_epochs):
            current_loss = 0

            # Train
            self.train()
            for i, data in enumerate(train_loader, 0):
                X_batch, y_batch = data
                optimizer.zero_grad(set_to_none=True)
                outputs = self(X_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            print(f'Epoch {epoch} | loss {current_loss/len(train_loader):0.3f}') 


    def predict_proba(self, X):
        X = process_X(X)
        self.eval()
        with torch.no_grad():
            predictions = torch.sigmoid(self(X))
            predictions = predictions.detach().numpy().ravel()
            return predictions
