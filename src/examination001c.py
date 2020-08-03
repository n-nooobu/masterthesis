import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt.modulate import prbs, Modulate
from pyopt import wgn
from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle, image_to_binary


ml.GPU_off()
ml.log_off()

image_path = glob.glob(os.path.join('./image/train/', '*.jpg'))
mdl = Modulate('RZ16QAM')

# image0 = cv2.imread(image_path[0])[::10, ::10]
# image_binary0 = image_to_binary(image0)
bitsq16 = prbs(N=16, itr=0)
# random0 = np.random.randint(0, 2, 100000)
sq0 = mdl.transform(bitsq16)
sgnl0 = wgn.Signal(seq=sq0, form='RZ16QAM', n=32, SNR=12)
sgnl0.add_wgn()
print(ml.evm_score(sgnl0.signal['input'], sgnl0.signal['wgn_signal']))

# image1 = cv2.imread(image_path[1])[::10, ::10]
# image_binary1 = image_to_binary(image1)
# bitsq17 = prbs(N=17, itr=0)
random1 = np.random.randint(0, 2, 100000)
sq1 = mdl.transform(random1)
sgnl1 = wgn.Signal(seq=sq1, form='RZ16QAM', n=32, SNR=12)
sgnl1.add_wgn()
print(ml.evm_score(sgnl1.signal['input'], sgnl1.signal['wgn_signal']))

image2 = cv2.imread(image_path[0])[::10, ::10]
image_binary2 = image_to_binary(image2)
sq2 = mdl.transform(image_binary2)
sgnl2 = wgn.Signal(seq=sq2, form='RZ16QAM', n=32, SNR=12)
sgnl2.add_wgn()
print(ml.evm_score(sgnl2.signal['input'], sgnl2.signal['wgn_signal']))


result = np.zeros((4, 15), dtype=float)
print('tap iteration START')
for i, tap in enumerate(tqdm(np.arange(1, 30, 2))):
    X0, y0 = ml.data_shaping_with_overlapping(sgnl0.signal['input'], sgnl0.signal['wgn_signal'], 29, tap)
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.3, random_state=1, stratify=y0)
    sc_y = StandardScaler()
    y_train_std0 = sc_y.fit_transform(y_train0)
    y_test_std0 = sc_y.fit_transform(y_test0)

    X1, y1 = ml.data_shaping_with_overlapping(sgnl1.signal['input'], sgnl1.signal['wgn_signal'], 29, tap)
    y_std1 = sc_y.transform(y1)

    X2, y2 = ml.data_shaping_with_overlapping(sgnl2.signal['input'], sgnl2.signal['wgn_signal'], 29, tap)
    y_std2 = sc_y.transform(y2)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(epochs=300, lr=0.001, log=False))
    pipe.fit(X_train0, y_train_std0)
    y_pred_std_train = pipe.predict(X_train0)
    result[0, i] = ml.evm_score(y_train_std0, y_pred_std_train)
    y_pred_std_test = pipe.predict(X_test0)
    result[1, i] = ml.evm_score(y_test_std0, y_pred_std_test)
    y_pred_std1 = pipe.predict(X1)
    result[2, i] = ml.evm_score(y_std1, y_pred_std1)
    y_pred_std2 = pipe.predict(X2)
    result[3, i] = ml.evm_score(y_std2, y_pred_std2)

# np.savetxt('result/result001c_06.csv', result, delimiter=',')

# result = np.loadtxt('result/result001c_06.csv', delimiter=',')

n = np.arange(1, 30, 2)
fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot(n, result[0], '.-', label='PRBS16 used to train(Trained on PRBS16(itr=1))')
line, = ax.plot(n, result[1], '.-', label='PRBS16 unused to train(Trained on PRBS16(itr=1))')
line, = ax.plot(n, result[2], '.-', label='ramdom(Trained on PRBS16(itr=1))')
line, = ax.plot(n, result[3], '.-', label='image0(Trained on PRBS16(itr=1))')
ax.legend(loc='lower right')
plt.xlabel('Number of Taps')
plt.ylabel('EVM[%]')
ax.set_ylim((18, 33))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()
