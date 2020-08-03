import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager
import optuna
from IPython.display import display
import matplotlib.pyplot as plt
from ipywidgets import Play, IntSlider, jslink, HBox, interactive_output

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyopt import machine_learning as ml
from pyopt.util import save_pickle, load_pickle


def val_once(tap, neuron, epochs, train_list, test_list, result):
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
    scores = result['cross_val_score']
    scores.append(evm1)
    result['cross_val_score'] = scores


def cross_val(tap, neuron, epochs, n_all, n_splits):
    print('neuron: ' + str(neuron) + ', epochs: ' + str(epochs) + '......START')
    manager = Manager()
    result = manager.dict({'cross_val_score': []})
    process_list = []
    kf = ml.KFold(n_all=n_all, n_splits=n_splits, shuffle=True)
    for train_list, test_list in zip(kf[0], kf[1]):
        process = Process(
            target=val_once,
            kwargs={'tap': tap, 'neuron': neuron, 'epochs': epochs,
                    'train_list': train_list, 'test_list': test_list, 'result': result})
        process.start()
        process_list.append(process)
    for process in tqdm(process_list):
        process.join()
    scores = result['cross_val_score']
    return np.mean(scores)


def objective(trial):
    neuron = trial.suggest_int('neuron', 10, 1000)
    epochs = trial.suggest_int('epochs', 50, 1000)
    tap = 1
    n_all = 10
    n_splits = 5
    score = cross_val(tap, neuron, epochs, n_all, n_splits)
    return score


if __name__ == '__main__':
    ml.GPU_off()
    ml.log_off()

    # 最適化を実行
    # study = optuna.create_study()
    study = load_pickle('results/optuna007.pickle')
    study.optimize(objective, n_trials=5)
    """
    # 結果をプロットする為に値を加工
    values = [each.value for each in study.trials]
    best_values = [np.min(values[:k + 1]) for k in range(len(values))]
    n = [each.params['neuron'] for each in study.trials]
    e = [each.params['epochs'] for each in study.trials]

    # プロット関数
    def f(k):
        plt.figure(figsize=(9, 4.5))
        ax = plt.subplot(121)
        #ax.set_xlim(np.min(x) - 5, np.max(x) + 5)
        #ax.set_ylim(np.min(values) - 300, np.max(values) + 300)
        ax.scatter(e[:k], n[:k], c=values[:k], cmap='Blues')
        ax.scatter(e[k-1], n[k-1])

        ax = plt.subplot(122)
        ax.plot(best_values[:k])
        ax.set_yscale('log')
        plt.show()

    # ipywidgetで表示
    play = Play(value=1, min=0, max=len(study.trials), step=1, interval=500, description='Press play')
    slider = IntSlider(min=0, max=len(study.trials))
    jslink((play, 'value'), (slider, 'value'))
    ui = HBox([play, slider])
    out = interactive_output(f, {'k': slider})
    display(ui, out)
    """