import os
import glob
import cv2
from multiprocessing import Process

from pyopt.modulate import prbs, Modulate, eightb_tenb, image_to_binary
from pyopt import transmission as tr
from pyopt.util import save_pickle, load_pickle


def image_transmission(process_index, target, space, code='8B10B'):
    """画像をbitに変換し長距離伝送シミュレーションを実行する"""
    image_path = glob.glob(os.path.join('../image/' + target + '/', '*.jpg'))
    image = cv2.imread(image_path[process_index])[::space, ::space].reshape(-1)
    if code == '8B10B':
        image_binary = eightb_tenb(image)
    else:
        image_binary = image_to_binary(image)

    mdl = Modulate('RZ16QAM')
    sq = mdl.transform(image_binary)

    sgnl = tr.Signal(seq=sq, form='RZ16QAM', PdBm=1)
    sgnl.transmission(Lmax=2500, ase=True)
    if code == '8B10B':
        path = '../data/input/' + target + '/' + target + '_' + str(process_index).zfill(5) + '_' + str(space) + '_8B10B.pickle'
    else:
        path = '../data/input/' + target + '/' + target + '_' + str(process_index).zfill(5) + '_' + str(space) + '.pickle'
    save_pickle(sgnl, path)


def loop_multiprocessing(start, end, target='train', space=10, code='8B10B'):
    process_list = []
    for i in range(start, end):
        process = Process(
            target=image_transmission,
            kwargs={'process_index': i, 'target': target, 'space': space, 'code': code})
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == '__main__':
    start = 100
    end = 130
    target = 'train_0'
    space = 10
    code = '8B10B'
    loop_multiprocessing(start, end, target, space, code)

