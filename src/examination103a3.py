import numpy as np
import matplotlib.pyplot as plt
import gc
import time
import datetime
from multiprocessing import Process, Manager

from sklearn.preprocessing import StandardScaler

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, Subset

from pyopt import machine_learning_pytorch as ml
from pyopt.util import save_pickle, load_pickle


def iteration_1step(process_index, tap, data, result):
    if result['evm_scores'][0, process_index] != 0:
        print('end')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)

    learning_rate = 0.001
    weight_decay = 0
    batch_size = 100
    num_workers = 0
    epochs = 500

    X_train = data['X_train'][:, int(len(data['X_train'][0]) / 2) - 1 - (tap - 1):
                              int(len(data['X_train'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    X_test0 = data['X_test0'][:, int(len(data['X_test0'][0]) / 2) - 1 - (tap - 1):
                              int(len(data['X_test0'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    X_test1 = data['X_test1'][:, int(len(data['X_test1'][0]) / 2) - 1 - (tap - 1):
                              int(len(data['X_test1'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]

    model = ml.PytorchNet(len(X_train[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train = ml.PytorchDatasets(X_train, data['y_train'])
    test0 = ml.PytorchDatasets(X_test0, data['y_test0'])
    test1 = ml.PytorchDatasets(X_test1, data['y_test1'])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test0_loader = DataLoader(test0, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test1_loader = DataLoader(test1, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(epochs):
        start_time = time.time()
        evm = 0
        train_losses = 0

        # === TRAIN ===
        # Sets the module in training mode.
        model.train()

        for X, y in train_loader:
            # Save them to device
            X = X.to(device)
            y = y.to(device)

            # Clear gradients first; very important, usually done BEFORE prediction
            optimizer.zero_grad()

            # Log Probabilities & Backpropagation
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # --- Save information after this batch ---
            # Save loss
            train_losses += loss.item()
            # Number of correct predictions
            evm += ml.evm_score(y.to('cpu').detach().numpy(), out.to('cpu').detach().numpy())

        # Compute Train EVM
        train_evm = evm / (len(X_train) / batch_size)

        # Compute time on Train
        duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]

        print('{} | Epoch: {}/{} | Loss: {:.4} | Train EVM: {:.3}'.format(duration, epoch + 1, epochs, train_losses,
                                                                          train_evm))

    # === EVAL ===
    # Sets the model in evaluation mode
    model.eval()

    # Create matrix to store evaluation predictions (for accuracy)
    #train_preds = torch.zeros(size=(len(X_train), 2), device=device, dtype=torch.float32)
    #test0_preds = torch.zeros(size=(len(X_test0), 2), device=device, dtype=torch.float32)
    #test1_preds = torch.zeros(size=(len(X_test1), 2), device=device, dtype=torch.float32)
    train_preds = np.zeros((len(X_train), 2), dtype=float)
    test0_preds = np.zeros((len(X_test0), 2), dtype=float)
    test1_preds = np.zeros((len(X_test1), 2), dtype=float)

    # Disables gradients (we need to be sure no optimization happens)
    with torch.no_grad():
        for k, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            train_preds[k * len(X): k * len(X) + len(X)] = pred.to('cpu').detach().numpy()

        for k, (X, y) in enumerate(test0_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test0_preds[k * len(X): k * len(X) + len(X)] = pred.to('cpu').detach().numpy()

        for k, (X, y) in enumerate(test1_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test1_preds[k * len(X): k * len(X) + len(X)] = pred.to('cpu').detach().numpy()

        # Compute evm
        train_evm = ml.evm_score(data['y_train'], train_preds)
        test0_evm = ml.evm_score(data['y_test0'], test0_preds)
        test1_evm = ml.evm_score(data['y_test1'], test1_preds)

    tmp = result['evm_scores']
    tmp[0, process_index] = train_evm
    tmp[2, process_index] = test0_evm
    tmp[4, process_index] = test1_evm
    result['evm_scores'] = tmp


def loop_multiprocessing(data, result):
    process_list = []
    tap_list = [i * 10 + 1 for i in range(20)]
    for i, tap in enumerate(tap_list):
        process = Process(
            target=iteration_1step,
            kwargs={'process_index': i, 'tap': tap, 'data': data, 'result': result})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    return result


if __name__ == '__main__':
    manager = Manager()
    # data = manager.dict()
    data = {}
    result = np.zeros((6, 20), dtype=float)

    number_of_train_data = 10
    X_train = np.array([])
    y_train = np.array([])
    evm_train = 0
    for train_idx in range(number_of_train_data):
        image = load_pickle('../data/input/train_0/train_0_' + str(train_idx).zfill(5) + '_10_8B10B.pickle')
        evm_train += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_train = np.append(X_train, X_tmp).reshape(-1, 191 * 2)
        y_train = np.append(y_train, y_tmp).reshape(-1, 2)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    data['X_train'] = np.array(X_train[:len(X_train) // 100 * 100], dtype=np.float32)
    data['y_train'] = np.array(y_train[:len(y_train) // 100 * 100], dtype=np.float32)
    result[1] = np.array([evm_train / number_of_train_data] * len(result[0]))

    number_of_test_data = 10
    X_test0 = np.array([])
    y_test0 = np.array([])
    evm_test0 = 0
    for test_idx in range(number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test0 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_test0 = np.append(X_test0, X_tmp).reshape(-1, 191 * 2)
        y_test0 = np.append(y_test0, y_tmp).reshape(-1, 2)
    X_test0 = sc_x.transform(X_test0)
    y_test0 = sc_y.transform(y_test0)
    data['X_test0'] = np.array(X_test0[:len(X_test0) // 100 * 100], dtype=np.float32)
    data['y_test0'] = np.array(y_test0[:len(y_test0) // 100 * 100], dtype=np.float32)
    result[3] = np.array([evm_test0 / number_of_test_data] * len(result[0]))

    X_test1 = np.array([])
    y_test1 = np.array([])
    evm_test1 = 0
    for test_idx in range(number_of_test_data, 2 * number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test1 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_test1 = np.append(X_test1, X_tmp).reshape(-1, 191 * 2)
        y_test1 = np.append(y_test1, y_tmp).reshape(-1, 2)
    X_test1 = sc_x.transform(X_test1)
    y_test1 = sc_y.transform(y_test1)
    data['X_test1'] = np.array(X_test1[:len(X_test1) // 100 * 100], dtype=np.float32)
    data['y_test1'] = np.array(y_test1[:len(y_test1) // 100 * 100], dtype=np.float32)
    result[5] = np.array([evm_test1 / number_of_test_data] * len(result[0]))

    del image, X_tmp, y_tmp
    gc.collect()

    result = manager.dict({'evm_scores': result})

    # result = np.loadtxt('../results/result103a_23.csv', delimiter=',')
    result = loop_multiprocessing(data, result)
    evm_scores = result['evm_scores']
    np.savetxt('../results/result103a_31.csv', evm_scores, delimiter=',')
