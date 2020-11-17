import sys
import os
import time
import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from pyopt.util import save_pickle, load_pickle


def data_shaping(input_signal, signal, max_tap, tap):
    """
    input_signal: 伝送前の信号
    signal: 伝送後の信号
    max_tap: 最大の同時入力シンボル数
    tap: 同時入力シンボル数

    signal = [x_0, x_1, ... , x_(n-1)]
      |
      |
      v
    x = [[x_0, x_1, ... , x_tap-1],
            [x_1, x_2, ..., x_tap],
                   .
                   .
                   .
            [x_(n-tap), x_(n-tap+1), ..., x(n-1)]]
      |
      |
      v
    x = [[i_0, q_0, i_1, q_1, ... , i_(tap-1), q_(tap-1)],
            [i_1, q_1, i_2, q_2, ... , i_tap, q_tap],
                   .
                   .
                   .
            [i_(n-tap), q_(n-tap), i_(n-tap+1), q_(n-tap+1), ..., i_(n-1), q_(n-1)]] (batch, input_dim) input_dim = tap * 2

    y  (batch, output_dim) output_dim = 2
    """

    x = np.zeros((len(input_signal) - (max_tap - 1), tap * 2), dtype=float)
    y = np.zeros((len(input_signal) - (max_tap - 1), 2), dtype=float)
    for i, j in enumerate(np.arange(int((max_tap - 1) / 2), len(input_signal) - int((max_tap - 1) / 2))):
        x[i, 0::2] = signal[j - int((tap - 1) / 2): j + int((tap - 1) / 2) + 1].real
        x[i, 1::2] = signal[j - int((tap - 1) / 2): j + int((tap - 1) / 2) + 1].imag
        y[i, 0] = input_signal[j].real
        y[i, 1] = input_signal[j].imag
    return x, y


class Dataset(data.Dataset):
    def __init__(self, x, y, mean, std):
        self.x, self.y, self.mean, self.std = x, y, mean, std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return torch.Tensor(x), torch.Tensor(y)


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neuron):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neuron)
        self.fc2 = nn.Linear(hidden_neuron, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evm_score(y_true, y_pred):
    if y_true.ndim == 2:
        y_true = y_true[:, 0] + 1j * y_true[:, 1]
        y_pred = y_pred[:, 0] + 1j * y_pred[:, 1]
    tmp = 0
    for i in range(len(y_pred)):
        tmp += abs(y_pred[i] - y_true[i]) ** 2 / abs(y_true[i]) ** 2
    evm = torch.sqrt(tmp / len(y_pred)) * 100
    return evm


def train_model(device, model, dataloaders_dict, criterion, optimizer, epochs, epochs_section=None):
    for epoch in range(epochs):
        if epochs_section is not None:
            epoch += epochs_section[0]
            end_epoch = epochs_section[1]
        else:
            end_epoch = epochs

        start_time = time.time()

        for phase in dataloaders_dict.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_evms = 0.0

            for x, y in dataloaders_dict[phase]:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x)
                    loss = criterion(outputs, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * x.size(0)
                    epoch_evms = (evm_score(y, outputs) / 100) ** 2 * x.size(0)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_evm = torch.sqrt(epoch_evms / len(dataloaders_dict[phase].dataset)) * 100

            duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
            print('{} | Epoch: {}/{} | {} Loss: {:.4} | EVM: {:.4}'.format(duration, epoch + 1, end_epoch, phase,
                                                                           epoch_loss, epoch_evm))
    return model


def ANN_train(tap):
    # tap = 1
    max_tap = 501
    batch_size = 100
    neuron = 300
    epochs = 1000
    lr = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)

    signal_type = 'image'

    form = 'RZ16QAM'  # 変調方式
    n = 32  # 1シンボルあたりのサンプリング数[/symbol]
    equalize = False  # 各シンボル数を均等にするか
    baudrate = 28  # ボーレート[GBaud]
    PdBm = 1  # 平均入力光パワー[dBm]
    Ledfa = 100  # EDFAスパン[km]
    stepedfa = 30  # SSFMの繰り返し計算ステップ数
    gamma = 1.4  # 非線形係数[/W/km]
    D = 16  # 分散パラメータ[ps/nm/km]
    Alpha = 0.16  # 伝送損失[dB/km]
    NF = 4  # ASE雑音指数[dB]
    Lmax = 500  # 伝送距離[km]
    ase = True  # ASE雑音を考慮するか

    if signal_type == 'prbs':
        N = 13  # PRBSの次数
        itr = 1  # PRBSの繰り返し回数
    elif signal_type == 'random':
        seed = 1234  # 乱数シード
        bit_num = 10000  # ビット長を指定
    elif signal_type == 'image':
        target_dir = 'train_0'
        step = 10  # =10 ---> (768, 1024) ---> (76, 102)
        image_number = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
        ebtb = True  # 8B10Bを行うか

    # prbs.csv or random.csv or image.csvをpandasで読み込む
    t_df_dir = '../data/input/'
    t_df = pd.read_csv(t_df_dir + signal_type + '.csv', index_col=0)

    # if ANN.csv がある: pandasで読み込む if ANN.csvがない: 新しいDataFrameを作る
    l_df_dir = '../data/params/ANN.csv'
    if os.path.exists(l_df_dir):
        l_df = pd.read_csv(l_df_dir, index_col=0)
    else:
        cols = ['tap', 'max_tap', 'batch_size', 'neuron', 'epochs', 'learning_rate', 'N', 'itr', 'seed', 'bit_num',
                'target_dir', 'step', 'image_number', 'ebtb', 'form', 'n', 'equalize', 'baudrate', 'PdBm', 'Ledfa',
                'stepedfa', 'gamma', 'D', 'Alpha', 'NF', 'Lmax', 'ase', 'params_path', 'train_samples']
        l_df = pd.DataFrame(index=[], columns=cols)
        l_df.to_csv(l_df_dir)

    # 指定した学習条件と伝送条件
    l_condition = 'tap==' + str(tap) + '&max_tap==' + str(max_tap) + '&batch_size==' + str(batch_size) + '&neuron==' + str(
        neuron) + '&learning_rate==' + str(lr)
    t_condition = 'form=="' + str(form) + '"&n==' + str(n) + '&equalize==' + str(equalize) + '&baudrate==' + str(
        baudrate) + '&PdBm==' + str(PdBm) + '&Ledfa==' + str(Ledfa) + '&stepedfa==' + str(stepedfa) + '&\
                            gamma==' + str(gamma) + '&D==' + str(D) + '&Alpha==' + str(Alpha) + '&NF==' + str(
        NF) + '&ase==' + str(ase)
    if signal_type == 'prbs':
        condition = 'N==' + str(N) + '&itr==' + str(itr)
        condition_list = [N, itr] + [None] * 6
    elif signal_type == 'random':
        condition = 'seed==' + str(seed) + '&bit_num==' + str(bit_num)
        condition_list = [None] * 2 + [seed, bit_num] + [None] * 4
    elif signal_type == 'image':
        condition = 'target_dir=="' + str(target_dir) + '"&step==' + str(step) + '&image_number==' + str(
            image_number[0]) + '&ebtb==' + str(ebtb)
        condition_list = [None] * 4 + [target_dir, step, image_number, ebtb]

    # prbs.csv or random.csv or image.csvにおいて、指定した伝送条件を見たす行を抜き出す
    t_query = t_df.query(condition + '&' + t_condition)

    # ANN.csvにおいて、指定した条件を満たす行だけqueryとして抜き出す
    l_query = l_df.query(l_condition + '&' + condition + '&' + t_condition + '&Lmax==' + str(Lmax))

    # 後で異なる値が代入されるもの以外をSeriesにしてしまう(epochs, params_path, train_samplesだけNone)
    sr = pd.Series(
        [tap, max_tap, batch_size, neuron, None, lr] + condition_list + [form, n, equalize, baudrate, PdBm, Ledfa, stepedfa,
                                                                         gamma, D, Alpha, NF, Lmax, ase, None, None],
        index=l_df.columns)

    # if epochsを含む指定された条件を満たす結果がある: 何もしない
    if len(l_query) > 0 and l_query['epochs'].max() >= epochs:
        print('指定された条件の学習結果はすでに存在します')
    else:
        # if epochs以外の指定された条件を満たす結果がある: パラメータを読み込む if ない: 新しくモデルを作成する
        if len(l_query) > 0:
            index = l_query['epochs'].idxmax()
            trained_epochs = l_query['epochs'][index]
            model = ANN(input_dim=tap * 2, output_dim=2, hidden_neuron=neuron).to(device)
            model.load_state_dict(torch.load(l_query['params_path'][index]))
        else:
            trained_epochs = 0
            model = ANN(input_dim=tap * 2, output_dim=2, hidden_neuron=neuron).to(device)

        # if prbs.csv or random.csv or image.csvに指定した伝送条件がない or Lmax以外は満たすがLmaxだけ指定した条件未満: 何もしない if ある: 続ける
        if len(t_query) == 0 or t_query.iloc[0]['Lmax'] < Lmax:
            print('指定された伝送条件の信号が存在しません')
        else:
            # 伝送信号を学習データに整形する
            sgnl_train = load_pickle(t_query.iloc[0]['data_path'])
            lc_train = sgnl_train.linear_compensation(Lmax, sgnl_train.signal['x_' + str(Lmax)])
            x_train, y_train = data_shaping(sgnl_train.signal['x_0'][n // 2::n], lc_train[n // 2::n], max_tap, tap)

            if signal_type == 'image' and len(image_number) > 1:
                for i in range(1, len(image_number)):
                    condition = 'target_dir=="' + str(target_dir) + '"&step==' + str(step) + '&image_number==' + str(
                        image_number[i]) + '&ebtb==' + str(ebtb)
                    t_query = t_df.query(condition + '&' + t_condition)
                    if len(t_query) == 0 or t_query.iloc[0]['Lmax'] < Lmax:
                        print('指定された伝送条件の信号が存在しません')
                        sys.exit()
                    sgnl_train = load_pickle(t_query.iloc[0]['data_path'])
                    lc_train = sgnl_train.linear_compensation(Lmax, sgnl_train.signal['x_' + str(Lmax)])
                    x_train_tmp, y_train_tmp = data_shaping(sgnl_train.signal['x_0'][n // 2::n], lc_train[n // 2::n],
                                                            max_tap, tap)
                    x_train = np.concatenate([x_train, x_train_tmp])
                    y_train = np.concatenate([y_train, y_train_tmp])

            train_samples = len(x_train)

            # 平均,標準偏差の計算
            mean = np.mean(x_train)
            std = np.std(x_train)

            # dataset, dataloaderの作成
            train_dataset = Dataset(x=x_train, y=y_train, mean=mean, std=std)
            train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaders_dict = {'train': train_dataloader}

            # 損失関数, オプティマイザの作成
            criterion = nn.MSELoss()
            optimizer = optim.Adam(params=model.parameters(), lr=lr)

            # モデルのトレーニング(50epochsずつ学習し、50epochsずつパラメータを保存する)
            for i in range((epochs - trained_epochs) // 50):
                model = train_model(device=device, model=model, dataloaders_dict=dataloaders_dict, criterion=criterion,
                                    optimizer=optimizer, epochs=50, epochs_section=[trained_epochs + i * 50, epochs])

                # 学習済みパラメータを保存し、ANN.csvに保存先を記入する
                l_df = pd.read_csv(l_df_dir, index_col=0)
                params_path = '../data/params/ANN/params_' + str(len(l_df)).zfill(10) + '.pth'
                torch.save(model.state_dict(), params_path)
                sr[4] = trained_epochs + (i + 1) * 50
                sr[-2] = params_path
                sr[-1] = train_samples
                l_df = l_df.append(sr, ignore_index=True)
                l_df.to_csv(l_df_dir)


def loop_multiprocessing(taps):
    process_list = []
    for tap in taps:
        process = Process(
            target=ANN_train,
            kwargs={'tap': tap})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == '__main__':
    taps = [1 + i * 10 for i in range(20)]
    loop_multiprocessing(taps)
