import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


def iteration_1step(process_index, tap, data, result):
    if result['evm_scores'][0, process_index] != 0:
        print('end')
        return

    X = data['X_train'][:, int(len(data['X_train'][0]) / 2) - 1 - (tap - 1):
                        int(len(data['X_train'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=300, epochs=500, lr=0.001, log=True))
    pipe.fit(X, data['y_train'])

    tmp = result['evm_scores']
    y_pred = pipe.predict(X)
    tmp[0, process_index] = ml.evm_score(data['y_train'], y_pred)
    X = data['X_test0'][:, int(len(data['X_test0'][0]) / 2) - 1 - (tap - 1):
                        int(len(data['X_test0'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    y_pred = pipe.predict(X)
    tmp[2, process_index] = ml.evm_score(data['y_test0'], y_pred)
    X = data['X_test1'][:, int(len(data['X_test1'][0]) / 2) - 1 - (tap - 1):
                        int(len(data['X_test1'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    y_pred = pipe.predict(X)
    tmp[4, process_index] = ml.evm_score(data['y_test1'], y_pred)
    X = data['X_N17'][:, int(len(data['X_N17'][0]) / 2) - 1 - (tap - 1):
                      int(len(data['X_N17'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    y_pred = pipe.predict(X)
    tmp[6, process_index] = ml.evm_score(data['y_N17'], y_pred)
    X = data['X_random'][:, int(len(data['X_random'][0]) / 2) - 1 - (tap - 1):
                         int(len(data['X_random'][0]) / 2) - 1 + 2 * (int((tap - 1) / 2) + 1)]
    y_pred = pipe.predict(X)
    tmp[8, process_index] = ml.evm_score(data['y_random'], y_pred)
    result['evm_scores'] = tmp


def loop_multiprocessing(data, result):
    manager = Manager()
    result = manager.dict({'evm_scores': result})
    process_list = []
    tap_list = [i * 5 + (1 - i % 2) for i in range(20)]
    for i, tap in enumerate(tap_list):
        process = Process(
            target=iteration_1step,
            kwargs={'process_index': i, 'tap': tap, 'data': data, 'result': result})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    return result


if __name__ == '__main__':
    data = {}
    result = np.zeros((10, 20), dtype=float)

    number_of_train_data = 1
    X_train = np.array([])
    y_train = np.array([])
    evm_train = 0
    for train_idx in range(number_of_train_data):
        image = load_pickle('../data/input/train_0/train_0_' + str(train_idx).zfill(5) + '_10_8B10B.pickle')
        evm_train += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_train = np.append(X_train, X_tmp).reshape(-1, 191 * 2)
        y_train = np.append(y_train, y_tmp).reshape(-1, 2)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    data['X_train'] = np.array(X_train, dtype=np.float32)
    data['y_train'] = np.array(y_train, dtype=np.float32)
    result[1] = np.array([evm_train / number_of_train_data] * len(result[0]))
    del X_train, y_train
    gc.collect()

    number_of_test_data = 10
    X_test0 = np.array([])
    y_test0 = np.array([])
    evm_test0 = 0
    for test_idx in range(number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test0 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_test0 = np.append(X_test0, X_tmp).reshape(-1, 191 * 2)
        y_test0 = np.append(y_test0, y_tmp).reshape(-1, 2)
    y_test0 = sc_y.transform(y_test0)
    data['X_test0'] = X_test0
    data['y_test0'] = y_test0
    result[3] = np.array([evm_test0 / number_of_test_data] * len(result[0]))
    del X_test0, y_test0
    gc.collect()

    X_test1 = np.array([])
    y_test1 = np.array([])
    evm_test1 = 0
    for test_idx in range(number_of_test_data, 2 * number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test1 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 501, 191)
        X_test1 = np.append(X_test1, X_tmp).reshape(-1, 191 * 2)
        y_test1 = np.append(y_test1, y_tmp).reshape(-1, 2)
    y_test1 = sc_y.transform(y_test1)
    data['X_test1'] = X_test1
    data['y_test1'] = y_test1
    result[5] = np.array([evm_test1 / number_of_test_data] * len(result[0]))
    del X_test1, y_test1
    gc.collect()

    N17 = load_pickle('../data/input/N17.pickle')
    evm_N17 = N17.cal_evm_pr(2500)
    X_tmp, y_tmp = ml.data_shaping_with_overlapping(N17.signal['x_0'], N17.signal['x_2500'], 501, 191)
    y_tmp = sc_y.transform(y_tmp)
    data['X_N17'] = X_tmp
    data['y_N17'] = y_tmp
    result[7] = np.array([evm_N17] * len(result[0]))

    random = load_pickle('../data/input/random00000.pickle')
    evm_random = random.cal_evm_pr(2500)
    X_tmp, y_tmp = ml.data_shaping_with_overlapping(random.signal['x_0'], random.signal['x_2500'], 501, 191)
    y_tmp = sc_y.transform(y_tmp)
    data['X_random'] = X_tmp
    data['y_random'] = y_tmp
    result[9] = np.array([evm_random] * len(result[0]))

    del image, N17, random, X_tmp, y_tmp
    gc.collect()

    # ml.GPU_restrict()
    ml.GPU_off()
    ml.log_off()

    # result = np.loadtxt('../results/result103a_30.csv', delimiter=',')
    result = loop_multiprocessing(data, result)
    evm_scores = result['evm_scores']
    # np.savetxt('../results/result103a_30.csv', evm_scores, delimiter=',')
