import numpy as np
import pickle


def fft(x):
    maxi = len(x)  # maxi eの配列長
    tmp = np.fft.fft(x)  # tmp eのフーリエ変換
    X = np.zeros(maxi, dtype=complex)
    for i in range(int(maxi / 2)):
        X[i] = tmp[i + int(maxi / 2)]
    for i in range(int(maxi / 2), int(maxi)):
        X[i] = tmp[i - int(maxi / 2)]

    return X


def ifft(X):
    maxi = len(X)
    tmp = np.zeros(maxi, dtype=complex)
    for i in range(int(maxi / 2)):
        tmp[i] = X[i + int(maxi / 2)]
    for i in range(int(maxi / 2), int(maxi)):
        tmp[i] = X[i - int(maxi / 2)]
    x = np.fft.ifft(tmp)

    return x


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
