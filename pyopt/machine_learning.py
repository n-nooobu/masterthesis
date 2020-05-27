import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


"""
class StandardScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.var_ = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.var_

    def inverse_transform(self, X):
        return X * self.var_ + self.mean_
"""


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


if __name__ == '__main__':

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

    """
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(StandardScaler)
    """
