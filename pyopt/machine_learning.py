import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
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

"""
def train_test_split(x, y, test_size):
    idx = int(len(x) * test_size)
    x_train = x[:-idx - 29:]
    x_test = x[-idx::]
    y_train = y[:-idx - 29:]
    y_test = y[-idx::]
    return x_train, x_test, y_train, y_test
"""


def accuracy_score_one_hot(y_true, y_pred):
    compare = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    return np.count_nonzero(compare) / len(y_true)


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


class ANNClass001(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=300, lr=0.001):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=len(X[0]), activation="relu"))
        self.model.add(Dense(3, activation="softmax"))
        # self.model.summary()
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=100, verbose=0, validation_split=0.1)
        return self

    def predict(self, X):
        return self.model.predict(X)


class ANNReg001(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=300, lr=0.001, log=True):
        self.epochs = epochs
        self.lr = lr
        self.log = log
        if self.log:
            self.verbose = 2
        else:
            self.verbose = 0
        self.model = Sequential()

    def fit(self, X, y):
        self.model.add(Dense(10, input_dim=len(X[0]), activation="relu"))
        self.model.add(Dense(2))
        if self.log:
            self.model.summary()
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=100, verbose=self.verbose, validation_split=0.1)
        return self

    def predict(self, X):
        return self.model.predict(X)


def GPU_restrict():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")


def GPU_off():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def log_off():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    def plot_decision_regions(X, y, classifier, one_hot=False, test_idx=None, resolution=0.02):
        # マーカーとカラーマップの準備
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # 決定領域のプロット
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # グリッドポイントの生成
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        # 各特徴量を1次元配列に変換して予測を実行
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        # 予測結果を元のグリッドポイントのデータサイズに変換
        Z2 = np.zeros(len(Z))
        if one_hot:
            for i in range(len(Z)):
                if np.argmax(Z[i]) == 0:
                    Z2[i] = 0
                elif np.argmax(Z[i]) == 1:
                    Z2[i] = 1
                else:
                    Z2[i] = 2
        Z = Z2.reshape(xx1.shape)
        # グリッドポイントの等高線のプロット
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        # 軸の範囲の設定
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # クラスごとにサンプルをプロット
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')

        # テストサンプルを目立たせる(点を○で表示)
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100,
                        label='test set')

    # データセット読み込み
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target  # Class labels: [0, 1, 2]
    #y = np.eye(len(np.unique(y)))[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    y_train_oh = np.eye(len(np.unique(y_train)))[y_train]
    y_test_oh = np.eye(len(np.unique(y_test)))[y_test]
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    y_combined_oh = np.vstack((y_train_oh, y_test_oh))

    # モデル構築・フィット・予測
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Perceptron
    pipe_ppn = make_pipeline(StandardScaler(),
                             Perceptron(eta0=0.1, random_state=1))
    # pipe_ppn.fit(X_train, y_train)

    pipe_ann = make_pipeline(StandardScaler(),
                             ANNClass001(epochs=10, lr=1.0))
    pipe_ann.fit(X_train, y_train_oh)


    """
    # グリッドサーチを行う
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    lr_range = [0.001, 0.01, 0.1, 1.0]
    epochs_range = [10, 30, 50, 100]
    param_grid = [{'ann001__lr': lr_range, 'ann001__epochs': epochs_range}]
    score_funcs = {'accuracy_one_hot': make_scorer(acuuracy_score_one_hot)}
    gs = GridSearchCV(estimator=pipe_ann,
                      param_grid=param_grid,
                      scoring=score_funcs['accuracy_one_hot'],
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train_oh)
    print(gs.best_score_)
    print(gs.best_params_)
    # scores = cross_val_score(gs, X_train, y_train_oh, scoring=score_funcs['accuracy_one_hot'], cv=5)
    """
    """
    # 交差検証を行う
    from sklearn.model_selection import cross_val_score
    scores_ppn = cross_val_score(estimator=pipe_ppn,
                                 X=X_train, y=y_train,
                                 cv=10, n_jobs=1)
    scores_ann = cross_val_score(estimator=pipe_ann,
                                 X=X_train, y=y_train_oh,
                                 cv=10, n_jobs=1)
    """

    # 決定領域をプロット
    plot_decision_regions(X=X_combined, y=y_combined, classifier=pipe_ann, one_hot=True, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

