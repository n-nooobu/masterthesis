import os
import glob
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from multiprocessing import Process

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt.modulate import prbs, Modulate, eightb_tenb, image_to_binary
from pyopt import transmission as tr
from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


"""長距離伝送シミュレーション"""

image_path = glob.glob(os.path.join('./image/train/', '*.jpg'))
image = cv2.imread(image_path[9])[::10, ::10].reshape(-1)
# image_binary = image_to_binary(image)
image_binary = eightb_tenb(image)
# bitsq = prbs(N=16, itr=0)
# random = np.random.randint(0, 2, 100000)
# random = np.array([randint(0, 1) for i in range(100000)])

mdl = Modulate('RZ16QAM')
sq = mdl.transform(image_binary)

# sgnl = tr.Signal(seq=sq, form='RZ16QAM', PdBm=1)
sgnl_ase = tr.Signal(seq=sq, form='RZ16QAM', PdBm=1)

# sgnl.transmission(Lmax=2500, ase=False)
# save_pickle(sgnl, 'dataset/trans_signal_image004_ver.1.0.pickle')

sgnl_ase.transmission(Lmax=2500, ase=True)
save_pickle(sgnl_ase, 'dataset/trans_signal_image009_ase_8b10b_ver.1.0.pickle')

"""
ml.GPU_off()
ml.log_off()

# random000 = load_pickle('dataset/trans_signal_random000_ver.1.0.pickle')
random000 = load_pickle('dataset/trans_signal_random000_ase_ver.1.0.pickle')
random001 = load_pickle('dataset/trans_signal_random001_ase_ver.1.0.pickle')
random002 = load_pickle('dataset/trans_signal_random002_ase_ver.1.0.pickle')

data = np.array([])
image010 = load_pickle('dataset/trans_signal_image010_ase_ver.1.0.pickle')
image011 = load_pickle('dataset/trans_signal_image011_ase_ver.1.0.pickle')
"""

"""
result = np.zeros((5, 15), dtype=float)
print('tap iteration START')
for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    X0, y0 = ml.data_shaping_with_overlapping(data0.signal['x_0'], data0.signal['x_2500'], 29, tap)
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.3, random_state=1, stratify=y0)
    sc_y = StandardScaler()
    y_train_std0 = sc_y.fit_transform(y_train0)
    y_test_std0 = sc_y.fit_transform(y_test0)

    X0_ase, y0_ase = ml.data_shaping_with_overlapping(data0_ase.signal['x_0'], data0_ase.signal['x_2500'], 29, tap)
    y_std0_ase = sc_y.transform(y0_ase)

    X1, y1 = ml.data_shaping_with_overlapping(data1_ase.signal['x_0'], data1_ase.signal['x_2500'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(data3_ase.signal['x_0'], data3_ase.signal['x_2500'], 29, tap)
    y_std2 = sc_y.transform(y2)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(epochs=300, lr=0.001, log=False))
    pipe.fit(X_train0, y_train_std0)
    y_pred_std0_train = pipe.predict(X_train0)
    result[0, i] = ml.evm_score(y_train_std0, y_pred_std0_train)
    y_pred_std0_test = pipe.predict(X_test0)
    result[1, i] = ml.evm_score(y_test_std0, y_pred_std0_test)
    y_pred_std0_ase = pipe.predict(X0_ase)
    result[2, i] = ml.evm_score(y_std0_ase, y_pred_std0_ase)
    y_pred_std1 = pipe.predict(X1)
    result[3, i] = ml.evm_score(y_std1, y_pred_std1)
    y_pred_std2 = pipe.predict(X2)
    result[4, i] = ml.evm_score(y_std2, y_pred_std2)

np.savetxt('result/result003a_01.csv', result, delimiter=',')

result = np.loadtxt('result/result003a_01.csv', delimiter=',')

n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='random0 used to train(Trained on random0)')
line, = ax.plot(n, result[1], '.-', label='random0 unused to train(Trained on random0)')
line, = ax.plot(n, result[2], '.-', label='random0_ase(Trained on random0)')
line, = ax.plot(n, result[3], '.-', label='random1_ase(Trained on random0)')
line, = ax.plot(n, result[4], '.-', label='image0_ase(Trained on random0)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
# ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""

"""
result = np.zeros((6, 15), dtype=float)
print('tap iteration START')
for i, tap in enumerate(tqdm(np.arange(1, 20, 2))):
    # X0, y0 = ml.data_shaping_with_overlapping(data3.signal['x_0'], data3.signal['x_2500'], 29, tap)
    X0 = np.array([])
    y0 = np.array([])
    for j in range(10):
        image = load_pickle('dataset/trans_signal_image00' + str(j) + '_ver.1.0.pickle')
        X0_tmp, y0_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 29, tap)
        X0 = np.append(X0, X0_tmp).reshape(-1, tap * 2)
        y0 = np.append(y0, y0_tmp).reshape(-1, 2)
    sc_y = StandardScaler()
    y_std0 = sc_y.fit_transform(y0)

    X1, y1 = ml.data_shaping_with_overlapping(image010.signal['x_0'], image010.signal['x_2500'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(image011.signal['x_0'], image011.signal['x_2500'], 29, tap)
    y_std2 = sc_y.transform(y2)

    X3, y3 = ml.data_shaping_with_overlapping(random000.signal['x_0'], random000.signal['x_2500'], 29, tap)
    y_std3 = sc_y.transform(y3)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(epochs=300, lr=0.001, log=False))
    pipe.fit(X0, y_std0)
    pipe.fit(X1, y_std1)
    y_pred_std0 = pipe.predict(X0)
    result[0, i] = ml.evm_score(y_std0, y_pred_std0)
    y_pred_std1 = pipe.predict(X1)
    result[1, i] = ml.evm_score(y_std1, y_pred_std1)
    y_pred_std2 = pipe.predict(X2)
    result[2, i] = ml.evm_score(y_std2, y_pred_std2)
    y_pred_std3 = pipe.predict(X3)
    result[3, i] = ml.evm_score(y_std3, y_pred_std3)

np.savetxt('result/result003a_04.csv', result, delimiter=',')

result = np.loadtxt('result/result003a_04.csv', delimiter=',')

n = np.arange(1, 20, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='image0~9(Trained on image0~9)')
line, = ax.plot(n, result[1], '.-', label='image10(Trained on image0~9)')
line, = ax.plot(n, result[2], '.-', label='image11(Trained on image0~9)')
line, = ax.plot(n, result[3], '.-', label='random(Trained on image0~9)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
# ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""