import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


def cross_val(tap, n_idx, neuron, e_idx, epochs, n_all, n_splits, result):
    print('neuron: ' + str(neuron) + ', epochs: ' + str(epochs) + '......START')
    scores = []
    kf = ml.KFold(n_all=n_all, n_splits=n_splits, shuffle=True)
    for train_list, test_list in tqdm(zip(kf[0], kf[1]), total=len(kf[0])):
        X0 = np.array([])
        y0 = np.array([])
        for train_idx in train_list:
            image = load_pickle('dataset/image' + str(train_idx).zfill(5) + '.pickle')
            X0_tmp, y0_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 29, tap)
            X0 = np.append(X0, X0_tmp).reshape(-1, tap * 2)
            y0 = np.append(y0, y0_tmp).reshape(-1, tap * 2)
        sc_y = StandardScaler()
        y0_std = sc_y.fit_transform(y0)

        X1 = np.array([])
        y1 = np.array([])
        for test_idx in test_list:
            val_image = load_pickle('dataset/image' + str(test_idx).zfill(5) + '.pickle')
            X1_tmp, y1_tmp = ml.data_shaping_with_overlapping(val_image.signal['x_0'], val_image.signal['x_2500'], 29, tap)
            X1 = np.append(X1, X1_tmp).reshape(-1, tap * 2)
            y1 = np.append(y1, y1_tmp).reshape(-1, tap * 2)
        y1_std = sc_y.transform(y1)

        pipe = make_pipeline(StandardScaler(),
                             ml.ANNReg001(neuron=neuron, epochs=epochs, lr=0.001, log=False))
        pipe.fit(X0, y0_std)
        y1_pred = pipe.predict(X1)
        evm1 = ml.evm_score(y1_std, y1_pred)
        scores.append(evm1)
    score_mean = np.mean(scores)
    tmp = result['evm_scores']
    tmp[n_idx, e_idx] = score_mean
    result['evm_scores'] = tmp
    if score_mean < result['best_evm_score']:
        result['best_evm_score'] = score_mean
        result['best_params'] = {'neuron': neuron, 'epochs': epochs}


def loop_multiprocessing(tap, neuron, epochs, n_all, n_splits):
    manager = Manager()
    result = manager.dict({'evm_scores': np.zeros((len(neuron), len(epochs)), dtype=float),
                           'best_evm_score': 1000})
    process_list = []
    for n_idx, n in enumerate(neuron):
        for e_idx, e in enumerate(epochs):
            process = Process(
                target=cross_val,
                kwargs={'tap': tap, 'n_idx': n_idx, 'neuron': n, 'e_idx': e_idx, 'epochs': e,
                        'n_all': n_all, 'n_splits': n_splits, 'result': result})
            process.start()
            process_list.append(process)
    for process in process_list:
        process.join()
    return result


if __name__ == '__main__':
    # ml.GPU_restrict()
    ml.GPU_off()
    ml.log_off()

    tap = 1
    neuron = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
    epochs = np.array([150, 200, 250, 300, 350, 400, 450, 500])
    n_all = 10
    n_splits = 5

    result = loop_multiprocessing(tap, neuron, epochs, n_all, n_splits)
