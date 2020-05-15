import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers



class StandardScaler:
    def __init__(self):
        None

    def fit(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean



def datashaping(signal):
    tmp_signal = signal.signal[int(signal.n / 2):: signal.n]
    tmp_input = signal.input[int(signal.n / 2):: signal.n]
    x = np.zeros((len(tmp_input) - (signal.max_tap - 1), signal.tap * 2), dtype=float)
    y = np.zeros((len(tmp_input) - (signal.max_tap - 1), 2), dtype=float)
    for i, j in enumerate(np.arange(int((signal.max_tap - 1) / 2), len(tmp_input) - int((signal.max_tap - 1) / 2))):
        x[i, 0::2] = tmp_signal[j - int((signal.tap - 1) / 2): j + int((signal.tap - 1) / 2) + 1].real
        x[i, 1::2] = tmp_signal[j - int((signal.tap - 1) / 2): j + int((signal.tap - 1) / 2) + 1].imag
        y[i, 0] = tmp_input[j].real
        y[i, 1] = tmp_input[j].imag
    return x, y

"""
def train_test_split(x, y, test_size):
    idx = int(len(x) * test_size)
    x_train = x[:-idx - 29:]
    x_test = x[-idx::]
    y_train = y[:-idx - 29:]
    y_test = y[-idx::]
    return x_train, x_test, y_train, y_test
"""

def cal_evm(x_in, x):
    x_in = x_in[:, 0] + 1j * x_in[:, 1]
    x = x[:, 0] + 1j * x[:, 1]
    tmp = 0
    for i in range(len(x)):
        tmp += abs(x[i] - x_in[i]) ** 2 / abs(x_in[i]) ** 2
    evm = np.sqrt(tmp / len(x)) * 100
    return evm


class ANN:
    def __init__(self):
        None

    def training(self, signal):
        x, y = datashaping(signal)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
        self.sc = StandardScaler()
        self.sc.fit(self.x_train)
        self.x_train_std = self.sc.transform(self.x_train)
        self.y_train_std = self.sc.transform(self.y_train)
        self.x_test_std = self.sc.transform(self.x_test)

        self.model = Sequential()
        self.model.add(Dense(10, input_dim=len(self.x_train_std[0]), activation="relu"))
        self.model.add(Dense(2))
        # self.model.summary()
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        history = self.model.fit(self.x_train_std, self.y_train_std, epochs=300, batch_size=100, verbose=0, validation_split=0.1)

    def predict(self, data):
        p_data = self.model.predict(data)
        p_data = self.sc.inverse_transform(p_data)
        return p_data
