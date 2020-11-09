import os
import glob
from multiprocessing import Process

import pandas as pd
import numpy as np
from PIL import Image

from pyopt import modulate as md
from pyopt import transmission as tr
from pyopt.util import save_pickle, load_pickle


def image_transmission(image_number):
    target_dir = 'train_0'
    step = 10  # =10 ---> (768, 1024) ---> (76, 102)
    # image_number = 0
    ebtb = True  # 8B10Bを行うか
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
    Lmax = 2500  # 伝送距離[km]
    ase = True  # ASE雑音を考慮するか

    df_dir = '../data/input/image.csv'
    if os.path.exists(df_dir):
        df = pd.read_csv(df_dir, index_col=0)
    else:
        cols = ['target_dir', 'step', 'image_number', 'ebtb',
                    'form', 'n', 'equalize', 'baudrate', 'PdBm', 'Ledfa', 'stepedfa', 'gamma', 'D', 'Alpha', 'NF', 'Lmax', 'ase', 'data_path', 'image_path']
        df = pd.DataFrame(index=[], columns=cols)
        df.to_csv(df_dir)

    condition = (df['target_dir']==target_dir) & (df['step']==step) & (df['image_number']==image_number) & (df['ebtb']==ebtb) & (df['form']==form) & \
                         (df['n']==n) & (df['equalize']==equalize) & (df['baudrate']==baudrate) & (df['PdBm']==PdBm) & (df['Ledfa']==Ledfa) & \
                         (df['stepedfa']==stepedfa) & (df['gamma']==gamma) & (df['D']==D) & (df['Alpha']==Alpha) & (df['NF']==NF) & (df['ase']==ase)

    if sum(condition) > 0:
        index = df[condition].index[0]
        if df['Lmax'][index] >= Lmax:
            print('指定された条件の伝送データはすでに存在します')
        else:
            data_path = df['data_path'][index]
            sgnl = load_pickle(data_path)
            sgnl.transmission(Lmax=Lmax, ase=ase)
            save_pickle(sgnl, data_path)

            df.loc[index, 'Lmax'] = Lmax
            df.to_csv(df_dir)
    else:
        image_dir = '../data/image/'
        image_path_list = sorted(glob.glob(os.path.join(image_dir + target_dir, '*.jpg')))
        image_path = image_path_list[image_number]
        img = Image.open(image_path)
        img_array = np.array(img)
        imgsq = img_array[::step, ::step].reshape(-1)
        if ebtb:
            imgsq_binary = md.eightb_tenb(imgsq)
        else:
            imgsq_binary = md.image_to_binary(imgsq)

        mdl = md.Modulate(form=form, n=n, equalize=equalize)
        modsq = mdl.transform(imgsq_binary)
        sgnl = tr.Signal(seq=modsq, form=form, n=n, baudrate=baudrate, PdBm=PdBm, Ledfa=Ledfa, stepedfa=stepedfa,
                                   gamma=gamma, D=D, Alpha=Alpha, NF=NF)
        sgnl.transmission(Lmax=Lmax, ase=ase)

        df = pd.read_csv(df_dir, index_col=0)
        data_path = '../data/input/image/image_' + str(len(df)).zfill(10) + '.pickle'
        save_pickle(sgnl, data_path)
        sr = pd.Series([target_dir, step, image_number, ebtb,
                                   form, n, equalize, baudrate, PdBm, Ledfa, stepedfa, gamma, D, Alpha, NF, Lmax, ase, data_path, image_path], index=df.columns)
        df = df.append(sr, ignore_index=True)
        df.to_csv(df_dir)


def loop_multiprocessing(start, end):
    process_list = []
    for image_number in range(start, end):
        process = Process(
            target=image_transmission,
            kwargs={'image_number': image_number})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == '__main__':
    for i in range(6):
        start = i * 25
        end = (i + 1) * 25
        loop_multiprocessing(start, end)
