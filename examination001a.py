import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyopt import wgn
from pyopt import machine_learning as ml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



"""result001a_01.pngの描画"""
"""
result = np.zeros((5, 15), dtype=float)

signal9 = wgn.Signal(type='Normal', form='RZ16QAM', N=9, n=32, itr=64, SNR=4)
wgn_signal9 = wgn.addwgn(signal9)
wgn_signal9.max_tap = 29
signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=16, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal11.tap = tap
    x, y = ml.datashaping(wgn_signal11)
    x_train11, x_test11, y_train11, y_test11 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train11)
    x_test_std11 = sc.transform(x_test11)

    wgn_signal9.tap = tap
    ann9 = ml.ANN()
    ann9.training(wgn_signal9)
    p_test9 = ann9.predict(ann9.x_test_std)
    evm = ml.cal_evm(ann9.y_test, p_test9)
    result[0, i] = evm
    p_test11 = ann9.predict(x_test_std11)
    evm = ml.cal_evm(y_test11, p_test11)
    result[1, i] = evm

signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=16, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29
signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal15.tap = tap
    x, y = ml.datashaping(wgn_signal15)
    x_train15, x_test15, y_train15, y_test15 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train15)
    x_test_std15 = sc.transform(x_test15)

    wgn_signal11.tap = tap
    ann11 = ml.ANN()
    ann11.training(wgn_signal11)
    p_test11 = ann11.predict(ann11.x_test_std)
    evm = ml.cal_evm(ann11.y_test, p_test11)
    result[2, i] = evm
    p_test15 = ann11.predict(x_test_std15)
    evm = ml.cal_evm(y_test15, p_test15)
    result[3, i] = evm

signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal15.tap = tap
    ann15 = ml.ANN()
    ann15.training(wgn_signal15)
    p_test15 = ann15.predict(ann15.x_test_std)
    evm = ml.cal_evm(ann15.y_test, p_test15)
    result[4, i] = evm

np.savetxt('result/result001.csv', result, delimiter=',')

# result = np.loadtxt('result/result001.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='PRBS9(Trained on PRBS9)')
line, = ax.plot(n, result[1], '.-', label='PRBS11(Trained on PRBS9)')
line, = ax.plot(n, result[2], '.-', label='PRBS11(Trained on PRBS11)')
line, = ax.plot(n, result[3], '.-', label='PRBS15(Trained on PRBS11)')
line, = ax.plot(n, result[4], '.-', label='PRBS15(Trained on PRBS15)')
ax.legend(loc='upper left')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((30, 90))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""



"""result001a_02.pngの描画"""

result = np.zeros((4, 15), dtype=float)

signal10 = wgn.Signal(type='Normal', form='RZ16QAM', N=10, n=32, itr=32, SNR=4)
wgn_signal10 = wgn.addwgn(signal10)
wgn_signal10.max_tap = 29
signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=16, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal11.tap = tap
    x, y = ml.datashaping(wgn_signal11)
    x_train11, x_test11, y_train11, y_test11 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train11)
    x_test_std11 = sc.transform(x_test11)

    wgn_signal10.tap = tap
    ann10 = ml.ANN()
    ann10.training(wgn_signal10)
    p_test10 = ann10.predict(ann10.x_test_std)
    evm = ml.cal_evm(ann10.y_test, p_test10)
    result[0, i] = evm
    p_test11 = ann10.predict(x_test_std11)
    evm = ml.cal_evm(y_test11, p_test11)
    result[1, i] = evm

signal13 = wgn.Signal(type='Normal', form='RZ16QAM', N=12, n=32, itr=4, SNR=4)
wgn_signal13 = wgn.addwgn(signal13)
wgn_signal13.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal13.tap = tap
    ann13 = ml.ANN()
    ann13.training(wgn_signal13)
    p_test13 = ann13.predict(ann13.x_test_std)
    evm = ml.cal_evm(ann13.y_test, p_test13)
    result[2, i] = evm

signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal15.tap = tap
    ann15 = ml.ANN()
    ann15.training(wgn_signal15)
    p_test15 = ann15.predict(ann15.x_test_std)
    evm = ml.cal_evm(ann15.y_test, p_test15)
    result[3, i] = evm

np.savetxt('result/result.csv', result, delimiter=',')

# result = np.loadtxt('result/result.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-s', color=[31/255, 119/255, 180/255], label='PRBS10(Trained on PRBS10)')
line, = ax.plot(n, result[1], '--s', color=[31/255, 119/255, 180/255], label='PRBS11(Trained on PRBS10)')
line, = ax.plot(n, result[2], '-o', color=[255/255, 127/255, 14/255], label='PRBS12(Trained on PRBS12)')
line, = ax.plot(n, result[3], '-^', color=[44/255, 160/255, 44/255], label='PRBS15(Trained on PRBS15)')
ax.legend(loc='upper left')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((40, 75))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
