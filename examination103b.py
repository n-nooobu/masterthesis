import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle

"""
tap = 1
epochs = np.array([150, 200, 250, 300, 350, 400, 450, 500])
neuron = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
list0 = [1, 8, 5, 3, 0, 9, 4, 7, 6, 2]
# list0 = [i for i in range(number)]
# list1 = random.sample(list0, 10)

result = {'evm_scores': np.zeros((8, 8), dtype=float),
          'best_evm_score': 100,
          'best_params': {'neuron': 0, 'epochs': 0}}

for e_idx, e in enumerate(epochs):
    for n_idx, n in enumerate(neuron):
        print('neuron: ' + str(n) + ', epochs: ' + str(e) + '......START')

        score_mean = np.mean(scores)
        evm[n_idx, e_idx] = score_mean
        if score_mean < best_score:
            best_score = score_mean
            best_params['neuron'] = n
            best_params['epochs'] = e
"""
tap = 1
neuron = 300
epochs = 200
val_list = [1, 8, 5, 3, 0, 9, 4, 7, 6, 2]
cv_repeat = 5
# def cross_val(tap, neuron, epochs, val_list, cv_repeat):
scores = []
for val_idx in tqdm(range(cv_repeat)):
    X0 = np.array([])
    y0 = np.array([])
    for j in range(len(val_list)):
        if j == val_list[val_idx * 2] or j == val_list[val_idx * 2 + 1]:
            continue
        image = load_pickle('dataset/image' + str(j).zfill(5) + '.pickle')
        X0_tmp, y0_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 29, tap)
        X0 = np.append(X0, X0_tmp).reshape(-1, tap * 2)
        y0 = np.append(y0, y0_tmp).reshape(-1, tap * 2)
    sc_y = StandardScaler()
    y0_std = sc_y.fit_transform(y0)

    val_image1 = load_pickle('dataset/image' + str(val_list[val_idx * 2]).zfill(5) + '.pickle')
    X1, y1 = ml.data_shaping_with_overlapping(val_image1.signal['x_0'], val_image1.signal['x_2500'], 29, tap)
    y1_std = sc_y.transform(y1)

    val_image2 = load_pickle('dataset/image' + str(val_list[val_idx * 2 + 1]).zfill(5) + '.pickle')
    X2, y2 = ml.data_shaping_with_overlapping(val_image2.signal['x_0'], val_image2.signal['x_2500'], 29, tap)
    y2_std = sc_y.transform(y2)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=neuron, epochs=epochs, lr=0.001, log=True))
    pipe.fit(X0, y0_std)
    y1_pred = pipe.predict(X1)
    evm1 = ml.evm_score(y1_std, y1_pred)
    y2_pred = pipe.predict(X2)
    evm2 = ml.evm_score(y2_std, y2_pred)
    scores.append((evm1 + evm2) / 2)

"""
if __name__ == '__main__':
    # ml.GPU_off()
    ml.log_off()
"""
