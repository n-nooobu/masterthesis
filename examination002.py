import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from pyopt import transmission as tr
from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle

from sklearn.model_selection import train_test_split

"""歪信号生成"""
# signal = tr.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, boudrate=28, Lmax=5000, PdBm=3, Ledfa=100,
#                   stepedfa=30, gamma=1.4, D=16, Alpha=0.16, NF=4)
# tr.transmission(signal)

# init_seq = load_pickle('dataset/init_seqs_sample_video001.pickle')
# signal = tr.Signal(form='RZ16QAM', init_seq=init_seq, n=32, itr=0, boudrate=28, Lmax=3000, PdBm=3, Ledfa=100,
#                   stepedfa=30, gamma=1.4, D=16, Alpha=0.16, NF=4)
# tr.transmission(signal)

"""result002_01.png, result002_03.pngの描画"""

"""tap1~29まで繰り返し"""
"""
result = np.zeros((7, 15), dtype=float)

signal9 = load_pickle('dataset/trans_signal_PRBS9.pickle')
signal9 = signal9['x_3000']
signal9.max_tap = 29
signal11 = load_pickle('dataset/trans_signal_PRBS11.pickle')
signal11 = signal11['x_3000']
signal11.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal11.tap = tap
    x, y = ml.datashaping(signal11)
    x_train11, x_test11, y_train11, y_test11 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train11)
    x_test_std11 = sc.transform(x_test11)

    for j in range(5):
        signal9.tap = tap
        ann9 = ml.ANN()
        ann9.training(signal9)
        p_test9 = ann9.predict(ann9.x_test_std)
        evm = ml.cal_evm(ann9.y_test, p_test9)
        result[0, i] += evm
        p_test11 = ann9.predict(x_test_std11)
        evm = ml.cal_evm(y_test11, p_test11)
        result[1, i] += evm

signal9 = load_pickle('dataset/trans_signal_PRBS9.pickle')
signal9 = signal9['x_3000']
signal9.max_tap = 29
signal11 = load_pickle('dataset/trans_signal_PRBS11.pickle')
signal11 = signal11['x_3000']
signal11.max_tap = 29
signal15 = load_pickle('dataset/trans_signal_PRBS15.pickle')
signal15 = signal15['x_3000']
signal15.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal9.tap = tap
    x, y = ml.datashaping(signal9)
    x_train9, x_test9, y_train9, y_test9 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train9)
    x_test_std9 = sc.transform(x_test9)

    signal15.tap = tap
    x, y = ml.datashaping(signal15)
    x_train15, x_test15, y_train15, y_test15 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train15)
    x_test_std15 = sc.transform(x_test15)

    for j in range(5):
        signal11.tap = tap
        ann11 = ml.ANN()
        ann11.training(signal11)
        p_test11 = ann11.predict(ann11.x_test_std)
        evm = ml.cal_evm(ann11.y_test, p_test11)
        result[2, i] += evm
        p_test9 = ann11.predict(x_test_std9)
        evm = ml.cal_evm(y_test9, p_test9)
        result[3, i] += evm
        p_test15 = ann11.predict(x_test_std15)
        evm = ml.cal_evm(y_test15, p_test15)
        result[4, i] += evm

signal11 = load_pickle('dataset/trans_signal_PRBS11.pickle')
signal11 = signal11['x_3000']
signal11.max_tap = 29
signal15 = load_pickle('dataset/trans_signal_PRBS15.pickle')
signal15 = signal15['x_3000']
signal15.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal11.tap = tap
    x, y = ml.datashaping(signal11)
    x_train11, x_test11, y_train11, y_test11 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train11)
    x_test_std11 = sc.transform(x_test11)

    for j in range(5):
        signal15.tap = tap
        ann15 = ml.ANN()
        ann15.training(signal15)
        p_test15 = ann15.predict(ann15.x_test_std)
        evm = ml.cal_evm(ann15.y_test, p_test15)
        result[5, i] += evm
        p_test11 = ann15.predict(x_test_std11)
        evm = ml.cal_evm(y_test11, p_test11)
        result[6, i] += evm

result /= 5
np.savetxt('result/result002.csv', result, delimiter=',')

# result = np.loadtxt('result/result002.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='PRBS9(Trained on PRBS9)')
line, = ax.plot(n, result[1], '.-', label='PRBS11(Trained on PRBS9)')
line, = ax.plot(n, result[2], '.-', label='PRBS11(Trained on PRBS11)')
line, = ax.plot(n, result[3], '.-', label='PRBS9(Trained on PRBS11)')
line, = ax.plot(n, result[4], '.-', label='PRBS15(Trained on PRBS11)')
line, = ax.plot(n, result[5], '.-', label='PRBS15(Trained on PRBS15)')
line, = ax.plot(n, result[6], '.-', label='PRBS11(Trained on PRBS15)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
# ax.set_ylim((30, 90))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""



"""result002_02.pngの描画"""
"""
signal11 = load_pickle('dataset/trans_signal_PRBS11.pickle')
signal11 = signal11['x_3000']
signal11.max_tap = 29

signal15 = load_pickle('dataset/trans_signal_PRBS15.pickle')
signal15 = signal15['x_3000']
signal15.max_tap = 29

signal15.tap = 15
x, y = ml.datashaping(signal15)
x_train15, x_test15, y_train15, y_test15 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
sc = ml.StandardScaler()
sc.fit(x_train15)
x_test_std15 = sc.transform(x_test15)

signal11.tap = 15
ann11 = ml.ANN()
ann11.training(signal11)
p_test11 = ann11.predict(ann11.x_test_std)
evm11 = ml.cal_evm(ann11.y_test, p_test11)
p_test15 = ann11.predict(x_test_std15)
evm15 = ml.cal_evm(y_test15, p_test15)

fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(signal11.signal.real, signal11.signal.imag, '.', color=[200/255, 200/255, 200/255], label='linear compensated')
line, = ax.plot(p_test15[:, 0], p_test15[:, 1], '.', label='Tested by PRBS15(Trained on PRBS11)')
line, = ax.plot(p_test11[:, 0], p_test11[:, 1], '.', label='Tested by PRBS11(Trained on PRBS11)')
ax.legend(loc='lower left')
ax.set_xlim((-180000, 180000))
ax.set_ylim((-200000, 150000))
plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                bottom=False,
                left=False,
                right=False,
                top=False)
ax.set_aspect('equal')
plt.show()
"""



"""result002_04.pngの描画"""

result = np.zeros((9, 15), dtype=float)

signal15 = load_pickle('dataset/trans_signal_PRBS15.pickle')
signal15 = signal15['x_3000']
signal15.max_tap = 29
signal17 = load_pickle('dataset/trans_signal_PRBS17.pickle')
signal17 = signal17['x_3000']
signal17.max_tap = 29
signalimage = load_pickle('dataset/trans_signal_sample_video001_01.pickle')
signalimage = signalimage['x_3000']
signalimage.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal17.tap = tap
    x, y = ml.datashaping(signal17)
    x_train17, x_test17, y_train17, y_test17 = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train17)
    x_test_std17 = sc.transform(x_test17)
    signalimage.tap = tap
    x, y = ml.datashaping(signalimage)
    x_trainimage, x_testimage, y_trainimage, y_testimage = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_trainimage)
    x_test_stdimage = sc.transform(x_testimage)

    for j in range(5):
        signal15.tap = tap
        ann15 = ml.ANN()
        ann15.training(signal15)
        p_test15 = ann15.predict(ann15.x_test_std)
        evm = ml.cal_evm(ann15.y_test, p_test15)
        result[0, i] += evm
        p_test17 = ann15.predict(x_test_std17)
        evm = ml.cal_evm(y_test17, p_test17)
        result[1, i] += evm
        p_testimage = ann15.predict(x_test_stdimage)
        evm = ml.cal_evm(y_testimage, p_testimage)
        result[2, i] += evm

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal15.tap = tap
    x, y = ml.datashaping(signal15)
    x_train15, x_test15, y_train15, y_test15 = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train15)
    x_test_std15 = sc.transform(x_test15)
    signalimage.tap = tap
    x, y = ml.datashaping(signalimage)
    x_trainimage, x_testimage, y_trainimage, y_testimage = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_trainimage)
    x_test_stdimage = sc.transform(x_testimage)

    for j in range(5):
        signal17.tap = tap
        ann17 = ml.ANN()
        ann17.training(signal17)
        p_test17 = ann17.predict(ann17.x_test_std)
        evm = ml.cal_evm(ann17.y_test, p_test17)
        result[3, i] += evm
        p_test15 = ann17.predict(x_test_std15)
        evm = ml.cal_evm(y_test15, p_test15)
        result[4, i] += evm
        p_testimage = ann17.predict(x_test_stdimage)
        evm = ml.cal_evm(y_testimage, p_testimage)
        result[5, i] += evm

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    signal15.tap = tap
    x, y = ml.datashaping(signal15)
    x_train15, x_test15, y_train15, y_test15 = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train15)
    x_test_std15 = sc.transform(x_test15)
    signal17.tap = tap
    x, y = ml.datashaping(signal17)
    x_train17, x_test17, y_train17, y_test17 = ml.train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = ml.StandardScaler()
    sc.fit(x_train17)
    x_test_std17 = sc.transform(x_test17)

    for j in range(5):
        signalimage.tap = tap
        annimage = ml.ANN()
        annimage.training(signalimage)
        p_testimage = annimage.predict(annimage.x_test_std)
        evm = ml.cal_evm(annimage.y_test, p_testimage)
        result[6, i] += evm
        p_test15 = annimage.predict(x_test_std15)
        evm = ml.cal_evm(y_test15, p_test15)
        result[7, i] += evm
        p_test17 = annimage.predict(x_test_std17)
        evm = ml.cal_evm(y_test17, p_test17)
        result[8, i] += evm

result /= 5
np.savetxt('result/result.csv', result, delimiter=',')

result = np.loadtxt('result/result.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-.s', color=[31/255, 119/255, 180/255], label='PRBS15(Trained on PRBS15)')
line, = ax.plot(n, result[1], '--s', color=[31/255, 119/255, 180/255], label='PRBS17(Trained on PRBS15)')
line, = ax.plot(n, result[2], '-s', color=[31/255, 119/255, 180/255], label='image(Trained on PRBS15)')
line, = ax.plot(n, result[3], '--^', color=[255/255, 127/255, 14/255], label='PRBS17(Trained on PRBS17)')
line, = ax.plot(n, result[4], '-.^', color=[255/255, 127/255, 14/255], label='PRBS15(Trained on PRBS17)')
line, = ax.plot(n, result[5], '-^', color=[255/255, 127/255, 14/255], label='image(Trained on PRBS17)')
line, = ax.plot(n, result[6], '-o', color=[44/255, 160/255, 44/255], label='image(Trained on image)')
line, = ax.plot(n, result[7], '-.o', color=[44/255, 160/255, 44/255], label='PRBS15(Trained on image)')
line, = ax.plot(n, result[8], '--o', color=[44/255, 160/255, 44/255], label='PRBS17(Trained on image)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
# ax.set_ylim((30, 90))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()

