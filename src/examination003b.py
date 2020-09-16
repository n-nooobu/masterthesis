import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


ml.GPU_off()
ml.log_off()

tap = 1

list0 = [n for n in range(10)]
epochs = np.array([150, 200, 250, 300, 350, 400, 450, 500])
neuron = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
"""
# neuron = np.logspace(3.4, 10, 8, base=2).astype(np.int32)
evm = np.zeros((8, 8), dtype=float)
best_score = 100
best_params = {'neuron': 0, 'epochs': 0}

for e_idx, e in enumerate(epochs):
    for n_idx, n in enumerate(neuron):
        print('neuron: ' + str(n) + ', epochs: ' + str(e) + '......START')
        list1 = random.sample(list0, 10)
        scores = []
        # for val_idx in tqdm(range(5)):
        X0 = np.array([])
        y0 = np.array([])
        for j in range(10):
            if j != list1[4 * 2] or j != list1[4 * 2 + 1]:
                image = load_pickle('dataset/trans_signal_image00' + str(j) + '_ase_8b10b_ver.1.0.pickle')
                X0_tmp, y0_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 29, tap)
                X0 = np.append(X0, X0_tmp).reshape(-1, 2)
                y0 = np.append(y0, y0_tmp).reshape(-1, 2)
        sc_y = StandardScaler()
        y_std0 = sc_y.fit_transform(y0)

        image1 = load_pickle('dataset/trans_signal_image00' + str(4 * 2) + '_ase_8b10b_ver.1.0.pickle')
        X1, y1 = ml.data_shaping_with_overlapping(image1.signal['x_0'], image1.signal['x_2500'], 29, tap)
        y_std1 = sc_y.transform(y1)

        image2 = load_pickle('dataset/trans_signal_image00' + str(4 * 2 + 1) + '_ase_8b10b_ver.1.0.pickle')
        X2, y2 = ml.data_shaping_with_overlapping(image2.signal['x_0'], image2.signal['x_2500'], 29, tap)
        y_std2 = sc_y.transform(y2)

        pipe = make_pipeline(StandardScaler(),
                             ml.ANNReg001(neuron=n, epochs=e, lr=0.001, log=False))
        pipe.fit(X0, y_std0)
        y_pred_std1 = pipe.predict(X1)
        evm0 = ml.evm_score(y_std1, y_pred_std1)
        y_pred_std2 = pipe.predict(X2)
        evm1 = ml.evm_score(y_std2, y_pred_std2)
        scores.append((evm0 + evm1)/2)

        score_mean = np.mean(scores)
        evm[n_idx, e_idx] = score_mean
        if score_mean < best_score:
            best_score = score_mean
            best_params['neuron'] = n
            best_params['epochs'] = e

np.savetxt('result/result003b_01.csv', evm, delimiter=',')
"""
evm = np.loadtxt('../results/result003b_01.csv', delimiter=',')


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel('epochs', fontsize=18)
ax.set_ylabel('neuron', fontsize=18)
flag = [True, True, True]
evm_sort = np.sort(evm.reshape(-1))
for i in range(8):
    for j in range(8):
        if evm[i][j] == evm_sort[10] and flag[0]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 25) * 900 / 7, label='{0}%'.format(np.round(evm[i][j], decimals=1)))
            flag[0] = False
        elif evm[i][j] == evm_sort[32] and flag[1]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 25) * 900 / 7, label='{0}%'.format(np.round(evm[i][j], decimals=1)))
            flag[1] = False
        elif evm[i][j] == evm_sort[-10] and flag[2]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 25) * 900 / 7, label='{0}%'.format(np.round(evm[i][j], decimals=1)))
            flag[2] = False
        else:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 25) * 900 / 7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=14, bbox_to_anchor=(1.0, 0.9), labelspacing=1.8, prop={'size': 13}, title="EVM[%]")
ax.xaxis.set_tick_params(labelsize=16, direction='in')
ax.yaxis.set_tick_params(labelsize=16, direction='in')
plt.subplots_adjust(left=0.13, bottom=0.15, right=0.78, top=0.98)
