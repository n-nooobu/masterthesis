import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


ml.GPU_off()
ml.log_off()

random000 = load_pickle('dataset/trans_signal_random000_ase_ver.1.0.pickle')

prbs16 = load_pickle('dataset/trans_signal_PRBS16_ase_ver.1.0.pickle')
prbs17 = load_pickle('dataset/trans_signal_PRBS17_ase_ver.1.0.pickle')

image000 = load_pickle('dataset/trans_signal_image000_ase_ver.1.0.pickle')
image006 = load_pickle('dataset/trans_signal_image006_ase_ver.1.0.pickle')
image009 = load_pickle('dataset/trans_signal_image009_ase_ver.1.0.pickle')
image000_8b10b = load_pickle('dataset/trans_signal_image000_ase_8b10b_ver.1.0.pickle')
image006_8b10b = load_pickle('dataset/trans_signal_image006_ase_8b10b_ver.1.0.pickle')
image009_8b10b = load_pickle('dataset/trans_signal_image009_ase_8b10b_ver.1.0.pickle')


"""result003a_03.png, result003a_04.png"""
"""
result = np.zeros((10, 15), dtype=float)
print('tap iteration START')
for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    X0, y0 = ml.data_shaping_with_overlapping(prbs16.signal['x_0'], prbs16.signal['x_2500'], 29, tap)
    sc_y = StandardScaler()
    y_std0 = sc_y.fit_transform(y0)

    X1, y1 = ml.data_shaping_with_overlapping(prbs17.signal['x_0'], prbs17.signal['x_2500'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(image000_8b10b.signal['x_0'], image000_8b10b.signal['x_2500'], 29, tap)
    y_std2 = sc_y.transform(y2)

    X3, y3 = ml.data_shaping_with_overlapping(image009_8b10b.signal['x_0'], image009_8b10b.signal['x_2500'], 29, tap)
    y_std3 = sc_y.transform(y3)

    X4, y4 = ml.data_shaping_with_overlapping(random000.signal['x_0'], random000.signal['x_2500'], 29, tap)
    y_std4 = sc_y.transform(y4)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=300, epochs=500, lr=0.001, log=False))
    pipe.fit(X0, y_std0)
    y_pred_std0 = pipe.predict(X0)
    result[0, i] = ml.evm_score(y_std0, y_pred_std0)
    result[1, i] = prbs16.cal_evm_pr(2500)
    y_pred_std1 = pipe.predict(X1)
    result[2, i] = ml.evm_score(y_std1, y_pred_std1)
    result[3, i] = prbs17.cal_evm_pr(2500)
    y_pred_std2 = pipe.predict(X2)
    result[4, i] = ml.evm_score(y_std2, y_pred_std2)
    result[5, i] = image000_8b10b.cal_evm_pr(2500)
    y_pred_std3 = pipe.predict(X3)
    result[6, i] = ml.evm_score(y_std3, y_pred_std3)
    result[7, i] = image009_8b10b.cal_evm_pr(2500)
    y_pred_std4 = pipe.predict(X4)
    result[8, i] = ml.evm_score(y_std4, y_pred_std4)
    result[9, i] = random000.cal_evm_pr(2500)

np.savetxt('result/result003a_04.csv', result, delimiter=',')

result = np.loadtxt('result/result003a_04.csv', delimiter=',')

n = np.arange(1, 30, 2)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-s', color=[31/255, 119/255, 180/255], label='PRBS16(ANN trained on PRBS16)')
line, = ax.plot(n, result[1], '--', color=[31/255, 119/255, 180/255], label='PRBS16(linear compensation and phase rotation)')
line, = ax.plot(n, result[2], '-s', color=[255/255, 127/255, 14/255], label='PRBS17(ANN trained on PRBS16)')
line, = ax.plot(n, result[3], '--', color=[255/255, 127/255, 14/255], label='PRBS17(linear compensation and phase rotation)')
line, = ax.plot(n, result[4], '-s', color=[44/255, 160/255, 44/255], label='image0(ANN trained on PRBS16)')
line, = ax.plot(n, result[5], '--', color=[44/255, 160/255, 44/255], label='image0(linear compensation and phase rotation)')
line, = ax.plot(n, result[6], '-s', color=[214/255, 39/255, 40/255], label='image9(ANN trained on PRBS16)')
line, = ax.plot(n, result[7], '--', color=[214/255, 39/255, 40/255], label='image9(linear compensation and phase rotation)')
line, = ax.plot(n, result[8], '-s', color=[180/255, 167/255, 27/255], label='random(ANN trained on PRBS16)')
line, = ax.plot(n, result[9], '--', color=[180/255, 167/255, 27/255], label='random(linear compensation and phase rotation)')
plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of input symbols')
plt.ylabel('EVM[%]')
# ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.subplots_adjust(left=0.05, bottom=0.10, right=0.60, top=0.95)
plt.show()
"""

"""result003a_05.png, result003a_07.png"""
"""
result = np.zeros((10, 15), dtype=float)
print('tap iteration START')
for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    X0, y0 = ml.data_shaping_with_overlapping(image000.signal['x_0'], image000.signal['x_2500'], 29, tap)
    sc_y = StandardScaler()
    y_std0 = sc_y.fit_transform(y0)

    X1, y1 = ml.data_shaping_with_overlapping(image006.signal['x_0'], image006.signal['x_2500'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(image009.signal['x_0'], image009.signal['x_2500'], 29, tap)
    y_std2 = sc_y.transform(y2)

    X3, y3 = ml.data_shaping_with_overlapping(random000.signal['x_0'], random000.signal['x_2500'], 29, tap)
    y_std3 = sc_y.transform(y3)

    X4, y4 = ml.data_shaping_with_overlapping(prbs16.signal['x_0'], prbs16.signal['x_2500'], 29, tap)
    y_std4 = sc_y.transform(y4)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=300, epochs=500, lr=0.001, log=False))
    pipe.fit(X0, y_std0)
    y_pred_std0 = pipe.predict(X0)
    result[0, i] = ml.evm_score(y_std0, y_pred_std0)
    result[1, i] = image000.cal_evm_pr(2500)
    y_pred_std1 = pipe.predict(X1)
    result[2, i] = ml.evm_score(y_std1, y_pred_std1)
    result[3, i] = image006.cal_evm_pr(2500)
    y_pred_std2 = pipe.predict(X2)
    result[4, i] = ml.evm_score(y_std2, y_pred_std2)
    result[5, i] = image009.cal_evm_pr(2500)
    y_pred_std3 = pipe.predict(X3)
    result[6, i] = ml.evm_score(y_std3, y_pred_std3)
    result[7, i] = random000.cal_evm_pr(2500)
    y_pred_std4 = pipe.predict(X4)
    result[8, i] = ml.evm_score(y_std4, y_pred_std4)
    result[9, i] = prbs16.cal_evm_pr(2500)

np.savetxt('result/result003a_05.csv', result, delimiter=',')
"""
result = np.loadtxt('result/result003a_07.csv', delimiter=',')

n = np.arange(1, 30, 2)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-s', color=[31/255, 119/255, 180/255], label='image0(ANN trained on image0)')
line, = ax.plot(n, result[1], '--', color=[31/255, 119/255, 180/255], label='image0(linear compensation and phase rotation)')
line, = ax.plot(n, result[2], '-s', color=[255/255, 127/255, 14/255], label='image6(ANN trained on image0)')
line, = ax.plot(n, result[3], '--', color=[255/255, 127/255, 14/255], label='image6(linear compensation and phase rotation)')
line, = ax.plot(n, result[4], '-s', color=[44/255, 160/255, 44/255], label='image9(ANN trained on image0)')
line, = ax.plot(n, result[5], '--', color=[44/255, 160/255, 44/255], label='image9(linear compensation and phase rotation)')
line, = ax.plot(n, result[6], '-s', color=[214/255, 39/255, 40/255], label='random(ANN trained on image0)')
line, = ax.plot(n, result[7], '--', color=[214/255, 39/255, 40/255], label='random(linear compensation and phase rotation)')
line, = ax.plot(n, result[8], '-s', color=[180/255, 167/255, 27/255], label='PRBS16(ANN trained on image0)')
line, = ax.plot(n, result[9], '--', color=[180/255, 167/255, 27/255], label='PRBS16(linear compensation and phase rotation)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, prop={'size': 15})
plt.xlabel('Number of Taps', fontsize=18)
plt.ylabel('EVM[%]', fontsize=18)
# ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.subplots_adjust(left=0.05, bottom=0.10, right=0.50, top=0.95)
plt.show()


"""result003a_06.png, result003a_08.png"""
"""
list0 = [1, 5, 6, 2, 7, 3, 8, 4, 0, 9]
result = np.zeros((10, 15), dtype=float)
print('tap iteration START')
for i, tap in tqdm(enumerate(tqdm(np.arange(1, 30, 2)))):
    X0 = np.array([])
    y0 = np.array([])
    evm0 = 0
    for j in range(8):
        image = load_pickle('dataset/trans_signal_image00' + str(list0[j]) + '_ase_ver.1.0.pickle')
        evm0 += image.cal_evm_pr(2500)
        X0_tmp, y0_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 29, tap)
        X0 = np.append(X0, X0_tmp).reshape(-1, tap * 2)
        y0 = np.append(y0, y0_tmp).reshape(-1, 2)
    sc_y = StandardScaler()
    y_std0 = sc_y.fit_transform(y0)

    X1, y1 = ml.data_shaping_with_overlapping(image000.signal['x_0'], image000.signal['x_2500'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(image009.signal['x_0'], image009.signal['x_2500'], 29, tap)
    y_std2 = sc_y.transform(y2)

    X3, y3 = ml.data_shaping_with_overlapping(random000.signal['x_0'], random000.signal['x_2500'], 29, tap)
    y_std3 = sc_y.transform(y3)

    X4, y4 = ml.data_shaping_with_overlapping(prbs16.signal['x_0'], prbs16.signal['x_2500'], 29, tap)
    y_std4 = sc_y.transform(y4)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=300, epochs=500, lr=0.001, log=False))
    pipe.fit(X0, y_std0)
    y_pred_std0 = pipe.predict(X0)
    result[0, i] = ml.evm_score(y_std0, y_pred_std0)
    result[1, i] = evm0 / 8
    y_pred_std1 = pipe.predict(X1)
    result[2, i] = ml.evm_score(y_std1, y_pred_std1)
    result[3, i] = image000.cal_evm_pr(2500)
    y_pred_std2 = pipe.predict(X2)
    result[4, i] = ml.evm_score(y_std2, y_pred_std2)
    result[5, i] = image009.cal_evm_pr(2500)
    y_pred_std3 = pipe.predict(X3)
    result[6, i] = ml.evm_score(y_std3, y_pred_std3)
    result[7, i] = random000.cal_evm_pr(2500)
    y_pred_std4 = pipe.predict(X4)
    result[8, i] = ml.evm_score(y_std4, y_pred_std4)
    result[9, i] = prbs16.cal_evm_pr(2500)

np.savetxt('result/result003a_06.csv', result, delimiter=',')

# result = np.loadtxt('result/result003a_05.csv', delimiter=',')

n = np.arange(1, 30, 2)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '-s', color=[31/255, 119/255, 180/255], label='image1~8(ANN trained on image1~8)')
line, = ax.plot(n, result[1], '--', color=[31/255, 119/255, 180/255], label='image1~8(linear compensation and phase rotation)')
line, = ax.plot(n, result[2], '-s', color=[255/255, 127/255, 14/255], label='image0(ANN trained on image1~8)')
line, = ax.plot(n, result[3], '--', color=[255/255, 127/255, 14/255], label='image0(linear compensation and phase rotation)')
line, = ax.plot(n, result[4], '-s', color=[44/255, 160/255, 44/255], label='image9(ANN trained on image1~8)')
line, = ax.plot(n, result[5], '--', color=[44/255, 160/255, 44/255], label='image9(linear compensation and phase rotation)')
line, = ax.plot(n, result[6], '-s', color=[214/255, 39/255, 40/255], label='random(ANN trained on image1~8)')
line, = ax.plot(n, result[7], '--', color=[214/255, 39/255, 40/255], label='random(linear compensation and phase rotation)')
line, = ax.plot(n, result[8], '-s', color=[180/255, 167/255, 27/255], label='prbs16(ANN trained on image1~8)')
line, = ax.plot(n, result[9], '--', color=[180/255, 167/255, 27/255], label='prbs16(linear compensation and phase rotation)')
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
# ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.subplots_adjust(left=0.05, bottom=0.10, right=0.60, top=0.95)
plt.show()
"""
