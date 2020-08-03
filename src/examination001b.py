import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyopt import wgn
from pyopt import machine_learning as ml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



"""PRBS10, 11, 15   result001b_01.pngの描画"""
"""
result = np.zeros((4, 15), dtype=float)

signal10 = wgn.Signal(type='Normal', form='RZ16QAM', N=10, n=32, itr=0, SNR=4)
wgn_signal10 = wgn.addwgn(signal10)
wgn_signal10.max_tap = 29
signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=0, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
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

signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=0, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
    wgn_signal11.tap = tap
    ann11 = ml.ANN()
    ann11.training(wgn_signal11)
    p_test11 = ann11.predict(ann11.x_test_std)
    evm = ml.cal_evm(ann11.y_test, p_test11)
    result[2, i] = evm

signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
    wgn_signal15.tap = tap
    ann15 = ml.ANN()
    ann15.training(wgn_signal15)
    p_test15 = ann15.predict(ann15.x_test_std)
    evm = ml.cal_evm(ann15.y_test, p_test15)
    result[3, i] = evm

np.savetxt('result/result001.csv', result, delimiter=',')

# result = np.loadtxt('result/result001.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='PRBS10(Trained on PRBS10)')
line, = ax.plot(n, result[1], '.-', label='PRBS11(Trained on PRBS10)')
line, = ax.plot(n, result[2], '.-', label='PRBS11(Trained on PRBS11)')
line, = ax.plot(n, result[3], '.-', label='PRBS15(Trained on PRBS15)')
ax.legend(loc='upper left')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((30, 130))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""


"""PRBS15, 16, 17    result001b_02.pngの描画"""
"""
result = np.zeros((4, 15), dtype=float)

signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29
signal16 = wgn.Signal(type='Normal', form='RZ16QAM', N=16, n=32, itr=0, SNR=4)
wgn_signal16 = wgn.addwgn(signal16)
wgn_signal16.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
    wgn_signal16.tap = tap
    x, y = ml.datashaping(wgn_signal16)
    x_train16, x_test16, y_train16, y_test16 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train16)
    x_test_std16 = sc.transform(x_test16)

    wgn_signal15.tap = tap
    ann15 = ml.ANN()
    ann15.training(wgn_signal15)
    p_test15 = ann15.predict(ann15.x_test_std)
    evm = ml.cal_evm(ann15.y_test, p_test15)
    result[0, i] = evm
    p_test16 = ann15.predict(x_test_std16)
    evm = ml.cal_evm(y_test16, p_test16)
    result[1, i] = evm

signal16 = wgn.Signal(type='Normal', form='RZ16QAM', N=16, n=32, itr=0, SNR=4)
wgn_signal16 = wgn.addwgn(signal16)
wgn_signal16.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
    wgn_signal16.tap = tap
    ann16 = ml.ANN()
    ann16.training(wgn_signal16)
    p_test16 = ann16.predict(ann16.x_test_std)
    evm = ml.cal_evm(ann16.y_test, p_test16)
    result[2, i] = evm

signal17 = wgn.Signal(type='Normal', form='RZ16QAM', N=17, n=32, itr=0, SNR=4)
wgn_signal17 = wgn.addwgn(signal17)
wgn_signal17.max_tap = 29

for i, tap in enumerate(np.arange(1, 30, 2)):
    wgn_signal17.tap = tap
    ann17 = ml.ANN()
    ann17.training(wgn_signal17)
    p_test17 = ann17.predict(ann17.x_test_std)
    evm = ml.cal_evm(ann17.y_test, p_test17)
    result[3, i] = evm

np.savetxt('result/result001.csv', result, delimiter=',')

result = np.loadtxt('result/result001.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='PRBS15(Trained on PRBS15)')
line, = ax.plot(n, result[1], '.-', label='PRBS16(Trained on PRBS15)')
line, = ax.plot(n, result[2], '.-', label='PRBS16(Trained on PRBS16)')
line, = ax.plot(n, result[3], '.-', label='PRBS17(Trained on PRBS17)')
ax.legend(loc='upper left')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((45, 70))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
"""



"""PRBS10, 13, 15, 17    result001b_03.pngの描画"""
"""
result = np.zeros((6, 15), dtype=float)

signal10 = wgn.Signal(type='Normal', form='RZ16QAM', N=10, n=32, itr=0, SNR=4)
wgn_signal10 = wgn.addwgn(signal10)
wgn_signal10.max_tap = 29
signal11 = wgn.Signal(type='Normal', form='RZ16QAM', N=11, n=32, itr=0, SNR=4)
wgn_signal11 = wgn.addwgn(signal11)
wgn_signal11.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal11.tap = tap
    x, y = ml.datashaping(wgn_signal11)
    x_train11, x_test11, y_train11, y_test11 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train11)
    x_test_std11 = sc.transform(x_test11)

    for j in range(5):
        wgn_signal10.tap = tap
        ann10 = ml.ANN()
        ann10.training(wgn_signal10)
        p_test10 = ann10.predict(ann10.x_test_std)
        evm = ml.cal_evm(ann10.y_test, p_test10)
        result[0, i] += evm
        p_test11 = ann10.predict(x_test_std11)
        evm = ml.cal_evm(y_test11, p_test11)
        result[1, i] += evm

signal12 = wgn.Signal(type='Normal', form='RZ16QAM', N=12, n=32, itr=0, SNR=4)
wgn_signal12 = wgn.addwgn(signal12)
wgn_signal12.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    for j in range(5):
        wgn_signal12.tap = tap
        ann12 = ml.ANN()
        ann12.training(wgn_signal12)
        p_test12 = ann12.predict(ann12.x_test_std)
        evm = ml.cal_evm(ann12.y_test, p_test12)
        result[2, i] += evm

signal13 = wgn.Signal(type='Normal', form='RZ16QAM', N=13, n=32, itr=0, SNR=4)
wgn_signal13 = wgn.addwgn(signal13)
wgn_signal13.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    for j in range(5):
        wgn_signal13.tap = tap
        ann13 = ml.ANN()
        ann13.training(wgn_signal13)
        p_test13 = ann13.predict(ann13.x_test_std)
        evm = ml.cal_evm(ann13.y_test, p_test13)
        result[3, i] += evm

signal15 = wgn.Signal(type='Normal', form='RZ16QAM', N=15, n=32, itr=0, SNR=4)
wgn_signal15 = wgn.addwgn(signal15)
wgn_signal15.max_tap = 29
signal16 = wgn.Signal(type='Normal', form='RZ16QAM', N=16, n=32, itr=0, SNR=4)
wgn_signal16 = wgn.addwgn(signal16)
wgn_signal16.max_tap = 29

for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    wgn_signal16.tap = tap
    x, y = ml.datashaping(wgn_signal16)
    x_train16, x_test16, y_train16, y_test16 = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(x_train16)
    x_test_std16 = sc.transform(x_test16)

    for j in range(5):
        wgn_signal15.tap = tap
        ann15 = ml.ANN()
        ann15.training(wgn_signal15)
        p_test15 = ann15.predict(ann15.x_test_std)
        evm = ml.cal_evm(ann15.y_test, p_test15)
        result[4, i] += evm
        p_test16 = ann15.predict(x_test_std16)
        evm = ml.cal_evm(y_test16, p_test16)
        result[5, i] += evm

result /= 5
np.savetxt('result/result.csv', result, delimiter=',')
"""
result = np.loadtxt('result/result001b_03.csv', delimiter=',')
n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-s', color=[31/255, 119/255, 180/255], label='PRBS10(Trained on PRBS10)')
line, = ax.plot(n, result[1], '--s', color=[31/255, 119/255, 180/255], label='PRBS11(Trained on PRBS10)')
line, = ax.plot(n, result[2], '-o', color=[255/255, 127/255, 14/255], label='PRBS12(Trained on PRBS12)')
line, = ax.plot(n, result[3], '-x', color=[44/255, 160/255, 44/255], label='PRBS13(Trained on PRBS13)')
line, = ax.plot(n, result[4], '-^', color=[214/255, 39/255, 40/255], label='PRBS15(Trained on PRBS15)')
line, = ax.plot(n, result[5], '--^', color=[214/255, 39/255, 40/255], label='PRBS16(Trained on PRBS15)')
ax.legend(loc='upper left')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((45, 135))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
