import numpy as np
import math
import torch
import torch.nn as nn
from .base import BaseNuisanceLearner


class _MLP(nn.Module):
    """
    Simple fully-connected network with ReLU activations.
    Layer sizes are specified dynamically.
    """

    def __init__(self, input_dim: int, hidden_sizes: list, output_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        in_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralNetLearner(BaseNuisanceLearner):
    """
    PyTorch neural network as a nuisance learner.
    Architecture is specified dynamically via hidden_sizes.
    Default: 64 -> 32 -> 1, ReLU, no output activation.
    Uses early stopping to avoid overfitting.
    """

    def __init__(
        self,
        hidden_sizes: list = [64, 32],
        lr: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 10,
        val_frac: float = 0.1,
        random_state: int = 66,
        batch_size: int = 64,
    ):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.batch_size = batch_size
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetLearner":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n, input_dim = X.shape

        n_val = max(1, int(n * self.val_frac))
        n_train = n - n_val
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        self.model = _MLP(input_dim, self.hidden_sizes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            idx = torch.randperm(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                loss = loss_fn(self.model(X_train_t[batch_idx]),
                               y_train_t[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            y_pred = self.model(X_t).squeeze(-1).numpy()
        return y_pred


class TunedNeuralNetLearner(BaseNuisanceLearner):
    """
    Neural Network with Farrell et al. (2021) architecture:
    - depth = floor(log(n))
    - width = floor(n^{1/(2+p/n)})
    - dropout for regularization
    - random validation split (shuffled)
    Reference: Farrell, Liang, Misra (2021), Econometrica.
    """

    def __init__(
        self,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 10,
        val_frac: float = 0.1,
        random_state: int = 66,
        batch_size: int = 64,
    ):
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.batch_size = batch_size
        self.model = None

    def _farrell_architecture(self, n: int, p: int) -> list:
        """Farrell et al. (2021) architecture formula."""
        depth = max(1, int(math.log(n)))
        width = max(8, int(n ** (1 / (2 + p / n))))
        return [width] * depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedNeuralNetLearner":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n, p = X.shape
        hidden_sizes = self._farrell_architecture(n, p)

        # random shuffle before train/val split
        idx = np.random.permutation(n)
        X, y = X[idx], y[idx]

        n_val = max(1, int(n * self.val_frac))
        n_train = n - n_val
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        self.model = _MLP(p, hidden_sizes, dropout=self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            idx_t = torch.randperm(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = idx_t[start:start + self.batch_size]
                loss = loss_fn(self.model(X_train_t[batch_idx]),
                               y_train_t[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            y_pred = self.model(X_t).squeeze(-1).numpy()
        return y_pred