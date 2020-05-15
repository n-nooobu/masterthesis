from pyopt.modulate import modulate
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Signal:
    def __init__(self, type, form, N=11, n=32, itr=0, SNR=4):
        self.type = type
        self.form = form
        self.N = N
        self.n = n
        self.itr = itr
        self.SNR = SNR  #[dB]
        self.init_seq, self.input = modulate(type=self.type, form=self.form, init_seq=None, N=self.N, n=self.n, itr=self.itr)
        self.signal = deepcopy(self.input)

    def display(self, signal):
        sampling_signal = signal[int(self.n / 2):: self.n]
        fig = plt.figure()
        ax = fig.add_subplot()
        line, = ax.plot(sampling_signal.real, sampling_signal.imag, '.')
        ax.xaxis.set_tick_params(direction='in')
        ax.yaxis.set_tick_params(direction='in')
        plt.show()


def addwgn(signal):
    wgn_signal = deepcopy(signal)
    N0 = 1 / (10 ** (wgn_signal.SNR / 10))
    noise = np.random.normal(0, np.sqrt(N0 / 2), len(wgn_signal.input)) + 1j \
        * np.random.normal(0, np.sqrt(N0 / 2), len(wgn_signal.input))
    wgn_signal.signal += noise
    return wgn_signal