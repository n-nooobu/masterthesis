# System
import os
import glob
import time
import datetime

# Basics
import numpy as np
import pandas as pd

# pyopt
import pyopt.machine_learning as ml
from pyopt.util import save_pickle, load_pickle

# PyTorch
import torch
from torchvision.transforms import Compose
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def batch_divide(max_tap=95, batch_size=1000, start=0, target_num=10, target_dir='train_0'):
    X_list = []
    y_list = []
    evm = 0
    path = sorted(glob.glob(os.path.join('data/input/' + target_dir + '/', '*.pickle')))
    for i in range(target_num):
        sgnl = load_pickle(path[start + i])
        evm += sgnl.cal_evm_pr(2500)
        X, y = ml.data_shaping_with_overlapping(sgnl.signal['x_0'], sgnl.signal['x_2500'], 501, max_tap)
        for j in range(len(X) // batch_size):
            X_batch = X[j * batch_size: (j + 1) * batch_size]
            y_batch = y[j * batch_size: (j + 1) * batch_size]
            save_pickle(X_batch, 'data/input/' + target_dir + '_batch/X_' + str(start + i).zfill(5) + '_' + str(j).zfill(5) + '.pickle')
            save_pickle(y_batch, 'data/input/' + target_dir + '_batch/y_' + str(start + i).zfill(5) + '_' + str(j).zfill(5) + '.pickle')
            X_list.append('data/input/' + target_dir + '_batch/X_' + str(start + i).zfill(5) + '_' + str(j).zfill(5) + '.pickle')
            y_list.append('data/input/' + target_dir + '_batch/y_' + str(start + i).zfill(5) + '_' + str(j).zfill(5) + '.pickle')
    df = pd.DataFrame({'X_dir': X_list, 'y_dir': y_list})
    evm /= target_num
    return df, evm


class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, dct):
        dct['X'] = (dct['X'] - self.mean) / self.std
        dct['y'] = (dct['y'] - self.mean) / self.std
        return dct


class NLCDataset(Dataset):
    def __init__(self, df, tap, transform=None):
        self.df, self.tap, self.transform = df, tap, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = load_pickle(self.df['X_dir'][idx])
        X = X[:, int(X.shape[1] / 2) - 1 - (self.tap - 1): int(X.shape[1] / 2) - 1 + 2 * (int((self.tap - 1) / 2) + 1)]  # 中央シンボルを中心にtap数シンボルを取り出す
        y = load_pickle(self.df['y_dir'][idx])
        dct = {'X': X, 'y': y}

        if self.transform:
            dct = self.transform(dct)

        return dct['X'], dct['y']


class NLCNet(nn.Module):
    def __init__(self, input_size):
        super(NLCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X


# ----- STATICS -----
batch_size1 = 1000
batch_size2 = 100
max_tap = 95
target_num1 = 10
target_num2 = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)


# 学習データ, テストデータをバッチごとにpickleで保存
train_df, train_prevm = batch_divide(max_tap=max_tap, batch_size=batch_size1, start=0, target_num=target_num1, target_dir='train_0_8B10B')
test0_df, test0_prevm = batch_divide(max_tap=max_tap, batch_size=batch_size1, start=0, target_num=target_num2, target_dir='test_8B10B')
test1_df, test1_prevm = batch_divide(max_tap=max_tap, batch_size=batch_size1, start=10, target_num=target_num2, target_dir='test_8B10B')
N17_df, N17_prevm = batch_divide(max_tap=max_tap, batch_size=batch_size2, start=0, target_num=1, target_dir='N17')
random_df, random_prevm = batch_divide(max_tap=max_tap, batch_size=batch_size2, start=0, target_num=1, target_dir='random')

print(len(train_df) * batch_size1)

# 平均の計算
value_sum = 0
value_num = 0
for i in range(len(train_df)):
    X = load_pickle(train_df['X_dir'][i])
    value_sum += np.sum(X)
    value_num += X.size
mean = value_sum / value_num

# 標準偏差の計算
dispersion_sum = 0
for i in range(len(train_df)):
    X = load_pickle(train_df['X_dir'][i])
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            dispersion_sum += (X[j, k] - mean) ** 2
std = np.sqrt(dispersion_sum / value_num)


def train(process_index, tap, result):
    print('started')
    if result[0, process_index] != 0:
        print('end')
        return result

    # ----- STATICS -----
    epochs = 300
    learning_rate = 0.001
    num_workers = 8

    model = NLCNet(tap * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train = NLCDataset(train_df, tap, transform=Compose([StandardScaler(mean, std)]))
    test0 = NLCDataset(test0_df, tap, transform=Compose([StandardScaler(mean, std)]))
    test1 = NLCDataset(test1_df, tap, transform=Compose([StandardScaler(mean, std)]))
    N17 = NLCDataset(N17_df, tap, transform=Compose([StandardScaler(mean, std)]))
    random = NLCDataset(random_df, tap, transform=Compose([StandardScaler(mean, std)]))

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=num_workers)
    test0_loader = DataLoader(test0, batch_size=1, shuffle=False, num_workers=num_workers)
    test1_loader = DataLoader(test0, batch_size=1, shuffle=False, num_workers=num_workers)
    N17_loader = DataLoader(N17, batch_size=1, shuffle=False, num_workers=num_workers)
    random_loader = DataLoader(random, batch_size=1, shuffle=False, num_workers=num_workers)

    # === EPOCHS ===
    for epoch in range(epochs):
        start_time = time.time()
        evm = 0
        train_losses = 0

        model.train()
        model.float()
        for X, y in train_loader:
            X = X.to(device).float()
            y = y.to(device).float()  # y.shape [1, 1000, 2]

            optimizer.zero_grad()

            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_losses += loss.item()
            evm += ml.evm_score(y.to('cpu').detach().numpy()[0], out.to('cpu').detach().numpy()[0])

        train_evm = evm / len(train_df)

        duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
        print('{} | Epoch: {}/{} | Loss: {:.4} | Train EVM: {:.3}'.format(duration, epoch + 1, epochs, train_losses,
                                                                          train_evm))

    # === EVAL ===
    model.eval()

    train_evm = 0
    test0_evm = 0
    test1_evm = 0
    N17_evm = 0
    random_evm = 0

    with torch.no_grad():
        for k, (X, y) in enumerate(train_loader):
            X = X.to(device).float()
            pred = model(X)
            train_evm += ml.evm_score(y.to('cpu').detach().numpy()[0], pred.to('cpu').detach().numpy()[0])

        for k, (X, y) in enumerate(test0_loader):
            X = X.to(device).float()
            pred = model(X)
            test0_evm += ml.evm_score(y.to('cpu').detach().numpy()[0], pred.to('cpu').detach().numpy()[0])

        for k, (X, y) in enumerate(test1_loader):
            X = X.to(device).float()
            pred = model(X)
            test1_evm += ml.evm_score(y.to('cpu').detach().numpy()[0], pred.to('cpu').detach().numpy()[0])

        for k, (X, y) in enumerate(N17_loader):
            X = X.to(device).float()
            pred = model(X)
            N17_evm += ml.evm_score(y.to('cpu').detach().numpy()[0], pred.to('cpu').detach().numpy()[0])

        for k, (X, y) in enumerate(random_loader):
            X = X.to(device).float()
            pred = model(X)
            random_evm += ml.evm_score(y.to('cpu').detach().numpy()[0], pred.to('cpu').detach().numpy()[0])

        train_evm /= len(train_df)
        test0_evm /= len(test0_df)
        test1_evm /= len(test1_df)
        N17_evm /= len(N17_df)
        random_evm /= len(random_df)

    result[0, process_index] = train_evm
    result[1, process_index] = train_prevm
    result[2, process_index] = test0_evm
    result[3, process_index] = test0_prevm
    result[4, process_index] = test1_evm
    result[5, process_index] = test1_prevm
    result[6, process_index] = N17_evm
    result[7, process_index] = N17_prevm
    result[8, process_index] = random_evm
    result[9, process_index] = random_prevm
    return result


# 最初だけ
result = np.zeros((10, 20), dtype=float)
np.savetxt('result211_002.csv', result, delimiter=',')

tap_list = [i * 5 + (1 - i % 2) for i in range(20)]
for i, tap in enumerate(tap_list):
    result = np.loadtxt('result211_002.csv', delimiter=',')
    retsult = train(i, tap, result)
    np.savetxt('result211_002.csv', retsult, delimiter=',')
    time.sleep(10)
