import os
import glob
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer

from pyopt.modulate import prbs, Modulate, eightb_tenb, image_to_binary
from pyopt import transmission as tr
from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle

"""
# 配列を生成し,16QAMに変調する
bitsq = prbs(N=15, itr=4)
random = np.random.randint(0, 2, 100000)
image_path = glob.glob(os.path.join('../image/train/', '*.jpg'))
image = cv2.imread(image_path[9])[::10, ::10].reshape(-1)
image_binary = image_to_binary(image)
image_8B10B = eightb_tenb(image)
mdl = Modulate('RZ16QAM')
sq = mdl.transform(image_binary)

sgnl = tr.Signal(seq=sq, form='RZ16QAM', PdBm=1)
sgnl.transmission(Lmax=2500, ase=True)

save_pickle(sgnl, '../data/input/train/train_00009_10.pickle')
"""

sgnl = load_pickle('../data/input/train_8B10B/train_00009_10_8B10B.pickle')
tr.display_constellation_color(sgnl.signal['x_500'][::10], sgnl.seq[16:: 32][::10], count=True)

"""
# ml.GPU_restrict()
ml.GPU_off()
ml.log_off()

sgnl = load_pickle('dataset/trans_signal_PRBS15_ver.1.0.pickle')
X, y = ml.data_shaping_with_overlapping(sgnl, L=4000, max_tap=29, tap=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
sc_y = StandardScaler()
y_train_std = sc_y.fit_transform(y_train)
y_test_std = sc_y.fit_transform(y_test)

pipe = make_pipeline(StandardScaler(),
                     ml.ANNReg001(epochs=300, lr=0.001, log=False))

pipe.fit(X_train, y_train_std)
y_pred_std = pipe.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred_std)
evm = ml.evm_score(y_test_std, y_pred_std)
tr.display_constellation(y_pred, dtype='array')

scores_ann = cross_val_score(estimator=pipe,
                             X=X_train, y=y_train_std,
                             scoring=make_scorer(ml.evm_score, greater_is_better=False),
                             cv=10, n_jobs=1)
"""
"""
epochs = np.array([150, 200, 250, 300, 350, 400, 450, 500])
neuron = np.logspace(1, 10, 8, base=2)
evm = [[1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 5, 5, 5, 5, 5], [2, 2, 2, 2, 2, 2, 2, 2], [5, 6, 3, 2, 7, 3, 2, 5],
                [1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 5, 5, 5, 5, 5], [2, 2, 2, 2, 2, 2, 2, 2], [5, 6, 3, 2, 7, 3, 2, 5]]

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel('epochs')
ax.set_ylabel('neuron')
flag = [True, True, True]
evm_sort = np.sort(evm.reshape(-1))
for i in range(8):
    for j in range(8):
        if evm[i][j] == evm_sort[10] and flag[0]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 1) * 900 / 7, label=str(evm[i][j]) + '%')
            flag[0] = False
        elif evm[i][j] == evm_sort[32] and flag[1]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 1) * 900 / 7, label=str(evm[i][j]) + '%')
            flag[1] = False
        elif evm[i][j] == evm_sort[-10] and flag[2]:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 1) * 900 / 7, label=str(evm[i][j]) + '%')
            flag[2] = False
        else:
            ax.scatter(epochs[j], neuron[i], c='darkblue', s=50 + (evm[i][j] - 1) * 900 / 7)
plt.legend(bbox_to_anchor=(1.4, 0.9), labelspacing=1.8, prop={'size': 20})
"""
