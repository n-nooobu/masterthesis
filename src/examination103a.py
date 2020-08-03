import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


def iteration_1step(process_index, tap, result):
    if result['evm_scores'][0, process_index] != 0:
        print('end')
        return

    number_of_train_data = 130
    X_train = np.array([])
    y_train = np.array([])
    evm_train = 0
    for train_idx in range(number_of_train_data):
        image = load_pickle('../data/input/train_0/train_0_' + str(train_idx).zfill(5) + '_10_8B10B.pickle')
        evm_train += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 49, tap)
        X_train = np.append(X_train, X_tmp).reshape(-1, tap * 2)
        y_train = np.append(y_train, y_tmp).reshape(-1, 2)
    sc_y = StandardScaler()
    y_train_std = sc_y.fit_transform(y_train)

    number_of_test_data = 10
    X_test0 = np.array([])
    y_test0 = np.array([])
    evm_test0 = 0
    for test_idx in range(number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test0 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 49, tap)
        X_test0 = np.append(X_test0, X_tmp).reshape(-1, tap * 2)
        y_test0 = np.append(y_test0, y_tmp).reshape(-1, 2)
    y_test0_std = sc_y.transform(y_test0)

    X_test1 = np.array([])
    y_test1 = np.array([])
    evm_test1 = 0
    for test_idx in range(number_of_test_data, 2 * number_of_test_data):
        image = load_pickle('../data/input/test/test_' + str(test_idx).zfill(5) + '_10_8B10B.pickle')
        evm_test1 += image.cal_evm_pr(2500)
        X_tmp, y_tmp = ml.data_shaping_with_overlapping(image.signal['x_0'], image.signal['x_2500'], 49, tap)
        X_test1 = np.append(X_test1, X_tmp).reshape(-1, tap * 2)
        y_test1 = np.append(y_test1, y_tmp).reshape(-1, 2)
    y_test1_std = sc_y.transform(y_test1)

    pipe = make_pipeline(StandardScaler(),
                         ml.ANNReg001(neuron=300, epochs=500, lr=0.001, log=False))
    pipe.fit(X_train, y_train_std)
    y_train_pred = pipe.predict(X_train)
    y_test0_pred = pipe.predict(X_test0)
    y_test1_pred = pipe.predict(X_test1)

    tmp = result['evm_scores']
    tmp[0, process_index] = ml.evm_score(y_train_std, y_train_pred)
    tmp[1, process_index] = evm_train / number_of_train_data
    tmp[2, process_index] = ml.evm_score(y_test0_std, y_test0_pred)
    tmp[3, process_index] = evm_test0 / number_of_test_data
    tmp[4, process_index] = ml.evm_score(y_test1_std, y_test1_pred)
    tmp[5, process_index] = evm_test1 / number_of_test_data
    result['evm_scores'] = tmp


def loop_multiprocessing(result=None):
    manager = Manager()
    if result is None:
        result = manager.dict({'evm_scores': np.zeros((6, 25), dtype=float)})
    else:
        result = manager.dict({'evm_scores': result})
    process_list = []
    tap_list = [i * 10 + 1 for i in range(20)]
    for i, tap in enumerate(np.arange(1, 50, 2)):
        process = Process(
            target=iteration_1step,
            kwargs={'process_index': i, 'tap': tap, 'result': result})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    return result


if __name__ == '__main__':

    ml.GPU_off()
    ml.log_off()

    result = np.loadtxt('../results/result103a_11.csv', delimiter=',')
    result = loop_multiprocessing(result)
    evm_scores = result['evm_scores']
    np.savetxt('../results/result103a_11.csv', evm_scores, delimiter=',')
    """
    result = np.loadtxt('../results/result103a_02.csv', delimiter=',')

    n = np.arange(1, 30, 2)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    line, = ax.plot(n, result[0], '-s', color=[31 / 255, 119 / 255, 180 / 255], label='train0~9(ANN trained on train0~9)')
    line, = ax.plot(n, result[1], '--', color=[31 / 255, 119 / 255, 180 / 255], label='train0~9(linear compensation and phase rotation)')
    line, = ax.plot(n, result[2], '-s', color=[255 / 255, 127 / 255, 14 / 255], label='test0(ANN trained on train0~9)')
    line, = ax.plot(n, result[3], '--', color=[255 / 255, 127 / 255, 14 / 255], label='test0(linear compensation and phase rotation)')
    line, = ax.plot(n, result[4], '-s', color=[44 / 255, 160 / 255, 44 / 255], label='test1(ANN trained on train0~9)')
    line, = ax.plot(n, result[5], '--', color=[44 / 255, 160 / 255, 44 / 255], label='test1(linear compensation and phase rotation)')
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('Number of Taps')
    plt.ylabel('EVM[%]')
    # ax.set_ylim((18, 33))
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    plt.subplots_adjust(left=0.05, bottom=0.10, right=0.64, top=0.95)
    plt.show()
    """