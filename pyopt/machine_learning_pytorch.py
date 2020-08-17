import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def evm_score(y_true, y_pred):
    if y_true.ndim == 2:
        y_true = y_true[:, 0] + 1j * y_true[:, 1]
        y_pred = y_pred[:, 0] + 1j * y_pred[:, 1]
    tmp = 0
    for i in range(len(y_pred)):
        tmp += abs(y_pred[i] - y_true[i]) ** 2 / abs(y_true[i]) ** 2
    evm = np.sqrt(tmp / len(y_pred)) * 100
    return evm


def data_shaping_with_overlapping(input, signal, max_tap, tap):
    X = np.zeros((len(input) - (max_tap - 1), tap * 2), dtype=float)
    y = np.zeros((len(input) - (max_tap - 1), 2), dtype=float)
    for i, j in enumerate(np.arange(int((max_tap - 1) / 2), len(input) - int((max_tap - 1) / 2))):
        X[i, 0::2] = signal[j - int((tap - 1) / 2): j + int((tap - 1) / 2) + 1].real
        X[i, 1::2] = signal[j - int((tap - 1) / 2): j + int((tap - 1) / 2) + 1].imag
        y[i, 0] = input[j].real
        y[i, 1] = input[j].imag
    return X, y


class PytorchDatasets(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        out_X = self.X[index]
        out_y = self.y[index]
        return out_X, out_y


class PytorchNet(nn.Module):
    def __init__(self, input_size):
        super(PytorchNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
