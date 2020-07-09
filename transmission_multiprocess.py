import os
import glob
import cv2
from multiprocessing import Process

from pyopt.modulate import prbs, Modulate, eightb_tenb, image_to_binary
from pyopt import transmission as tr
from pyopt.util import save_pickle, load_pickle


def image_transmission(process_index):
    """画像をbitに変換し長距離伝送シミュレーションを実行する"""
    image_path = glob.glob(os.path.join('./image/train/', '*.jpg'))
    image = cv2.imread(image_path[process_index])[::10, ::10].reshape(-1)
    # image_binary = image_to_binary(image)
    image_binary = eightb_tenb(image)

    mdl = Modulate('RZ16QAM')
    sq = mdl.transform(image_binary)

    sgnl = tr.Signal(seq=sq, form='RZ16QAM', PdBm=1)
    sgnl.transmission(Lmax=2500, ase=True)
    save_pickle(sgnl, 'dataset/image' + str(process_index).zfill(5) + '.pickle')


def loop_multiprocessing(start, end):
    process_list = []
    for i in range(start, end):
        process = Process(
            target=image_transmission,
            kwargs={'process_index': i})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == '__main__':
    start = 10
    end = 30
    loop_multiprocessing(start, end)

