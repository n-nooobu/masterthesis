import numpy as np
import pickle
from tqdm import tqdm


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


def image_to_binary(image):
    image_decimal = image.reshape(-1)
    image_binary = np.zeros(len(image_decimal) * 8, dtype=int)
    print('image to binary START')
    for byte in tqdm(range(len(image_decimal))):
        for bit in range(8):
            image_binary[byte * 8 + bit] = min(1 << (7 - bit) & image_decimal[byte], 1)
    return image_binary
