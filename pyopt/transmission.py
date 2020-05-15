from pyopt.modulate import modulate
from pyopt.util import fft, ifft, save_pickle, load_pickle
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append('C:\\Users\kamiya\AppData\Local\Programs\Python\Python36\Lib\site-packages')
from tqdm import tqdm


class Signal:
    def __init__(self, type='Normal', form='RZ16QAM', init_seq=None, N=11, n=32, itr=0, boudrate=28, Lmax=1000, PdBm=0, Ledfa=100, stepedfa=30,
                 gamma=1.4, D=16, Alpha=0.16, NF=4):
        self.type = type
        self.form = form
        self.N = N
        self.n = n
        self.itr = itr
        self.boudrate = boudrate
        self.Lmax = Lmax
        self.Lnow = 0
        self.PdBm = PdBm
        self.Ledfa = Ledfa
        self.stepedfa = stepedfa
        self.gamma = gamma
        self.D = D
        self.Alpha = Alpha
        self.NF = NF

        """固定パラメータ"""
        self.bitratelist = {'NRZQPSK': self.boudrate * 2,
                            'RZQPSK': self.boudrate * 2,
                            'NRZ16QAM': self.boudrate * 4,
                            'RZ16QAM': self.boudrate * 4}
        self.bitrate = self.bitratelist[self.form]  # ビットレート[Gbps]
        self.B = self.boudrate * 2 * 10 ** 9  # 帯域幅[bit/s]
        self.c = 3 * 10 ** 5  # ファイバ中光速[km/sec]
        self.Lambda_c = 1550  # キャリア波長[nm]
        self.f_c = self.c * 10 ** 3 / (self.Lambda_c * 10 ** -9)  # キャリア周波数[Hz]
        self.Omega_c = 2 * np.pi * self.c * 10 ** 3 / self.Lambda_c  # キャリア光角周波数[G rad/s]
        self.nc = 1.46  # コア屈折率
        self.a = 5 * 10 ** -6  # コア半径[m]
        self.As = np.pi * self.a ** 2  # コア面積[m^2]
        self.Xi = 1.5  # コア面積補正
        self.B2 = (-self.D * self.Lambda_c ** 2 * 10 ** -6) / (2 * np.pi * self.c)  # 2次分散[ns^2/km]
        self.G = 10 ** (self.Alpha * self.Ledfa / 10)  # EDFA利得
        self.h = 6.626 * 10 ** -34  # h[j*s] プランク定数
        self.n2 = self.gamma * 10 ** -3 * self.Lambda_c * 10 ** -9 * self.nc * self.Xi * self.As / (120 * np.pi ** 2)
        self.nsp = 0.5 * 10 ** (self.NF / 10)

        """初期信号"""
        self.PW = 10 ** (self.PdBm / 10 - 3)  # 光電力[W]
        self.Phi_p = np.sqrt((120 * np.pi / self.nc) * self.PW / (self.Xi * self.As))  # 光電界強度[V/m]
        if init_seq is not None:
            self.init_seq, self.input = modulate(form=self.form, init_seq=init_seq, N=self.N, n=self.n, itr=self.itr)
        else:
            self.init_seq, self.input = modulate(type=self.type, form=self.form, N=self.N, n=self.n, itr=self.itr)
        self.input *= self.Phi_p
        self.signal = deepcopy(self.input)

        """軸生成"""
        self.length = len(self.input)
        self.Frequency = self._frequency()  # 周波数軸[GHz]
        self.Lambda = self._wavelength()  # 波長軸[nm]
        self.Omega = self._omegafrequency() * 10 ** -9  # 光角周波数軸[G rad]

    def _timeslot(self):  # Ts[ps/bit] bitあたりの時間長
        Ts = 1 / (self.bitrate * 10 ** 9) * 10 ** 12
        return Ts

    def _frequency(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        Ts = self._timeslot()  # Ts[ps] 1タイムスロット(n点)の時間
        dt = Ts / self.n  # dt[ps] 1点の時間
        fc = 3 * 10 ** 8 / (self.Lambda_c * 10 ** -9) * 10 ** -9  # fc[GHz]
        df = 1 / (dt * (10 ** -12) * self.length) * 10 ** -9  # df[GHz]
        fwidth = 1 / (2 * dt) * 10 ** 3
        Frequency = np.zeros(self.length, dtype=float)
        for i in range(self.length):
            Frequency[i] = fc + (fwidth - df * i)  # f[GHz]
        return Frequency

    def _wavelength(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        fr = self._frequency()  # f[GHz] 周波数軸
        Lambda = 3 * 10 ** 8 / (fr * 10 ** 9) * 10 ** 9  # Lambda[nm] 波長軸
        return Lambda

    def _omegafrequency(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        fr = self._frequency()  # f[GHz] 周波数軸
        Omega = 2 * np.pi * fr * 10 ** 9  # Omega[rad] 角周波数軸
        return Omega

    def phase_rotation_theo(self):
        Alpha2 = self.Alpha * 10 ** -4 * np.log(10)
        pr_theo = 2 * np.pi * self.n2 / (self.Lambda_c * 10 ** -9) \
                    * (1 - np.exp(-Alpha2 * self.Ledfa * 10 ** 3)) / Alpha2 \
                    * self.Phi_p ** 2 * int(self.Lnow / self.Ledfa) * 180/np.pi
        """if self.form[0] == 'R':
            pr_theo /= 2"""
        return pr_theo

    def phase_rotation_simu(self):
        tmp = np.zeros((2, 2), dtype=complex)
        for i in range(len(self.input)):
            if self.input[i].real > 0 and self.input[i].imag > 0:
                tmp = np.append(tmp, np.array([self.input[i], self.signal[i]]).reshape(1, 2), axis=0)
        tmp = tmp[2::]
        r_in = np.angle(tmp[:, 0]) * 180 / np.pi
        r = np.angle(tmp[:, 1]) * 180 / np.pi
        pr_simu = np.mean(-(r - r_in))
        return pr_simu

    def cal_evm(self):
        tmp = 0
        for i in range(len(self.signal)):
            tmp += abs(self.signal[i] - self.input[i]) ** 2 / abs(self.input[i]) ** 2
        evm = np.sqrt(tmp / len(self.signal)) * 100
        return evm

    def cal_evm_pr(self):
        pr_theo = self.phase_rotation_theo()
        signal = self.signal * np.exp(1j * pr_theo * np.pi / 180)
        tmp = 0
        for i in range(len(signal)):
            tmp += abs(signal[i] - self.input[i]) ** 2 / abs(self.input[i]) ** 2
        evm_pr = np.sqrt(tmp / len(signal)) * 100
        return evm_pr

    def cal_evm_min(self):
        evm_min = 200
        i_min = 0
        for i in range(360):
            signal = self.signal * np.exp(1j * i * np.pi/180)
            tmp = 0
            for j in range(len(signal)):
                tmp += abs(signal[j] - self.input[j]) ** 2 / abs(self.input[j]) ** 2
            evm = np.sqrt(tmp / len(signal)) * 100
            if evm < evm_min:
                evm_min = evm
                i_min = i
        return evm_min, i_min

    def cal_ber(self):
        evm = self.cal_evm()
        M = {'NRZQPSK': 4,
             'RZQPSK': 4,
             'NRZ16QAM': 16,
             'RZ16QAM': 16}
        ber = (1 - M[self.form] ** (-1 / 2)) / (1 / 2 * np.log2(M[self.form])) \
            * special.erfc(np.sqrt(3 / 2 / (M[self.form] - 1) / (evm / 100) ** 2))
        return ber

    def cal_qfac(self):
        ber = self.cal_ber()
        q = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * ber))
        return q

    def cal_snr(self):
        evm = self.cal_evm()
        snr = 10 * np.log10(1 / (evm / 100) ** 2)
        return snr

    def display(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        line, = ax.plot(self.signal.real, self.signal.imag, '.')
        #ax.legend()
        #ax.set_xlim((-150000, 150000))
        #ax.set_ylim((-150000, 150000))
        ax.xaxis.set_tick_params(direction='in')
        ax.yaxis.set_tick_params(direction='in')
        plt.show()


def transmission(signal):
    trans_signal = {}
    for i in tqdm(range(int(signal.Lmax / signal.Ledfa))):
        signal.signal = _add_nonlinear_distortion(signal, signal.signal)  # 線形歪,非線形歪,減衰を同時に加える
        signal.signal *= 10 ** (signal.Alpha * signal.Ledfa / 10)  # EDFAによる増幅を行う
        signal.signal = _add_ase_noise(signal, signal.signal)  # 増幅と同時にASE雑音を加える
        signal.Lnow += signal.Ledfa  # 伝送済み距離を更新する
        lc_signal = deepcopy(signal)
        lc_signal = linear_compensation(lc_signal)
        lc_signal.input = lc_signal.input[int(signal.n / 2):: signal.n]
        lc_signal.signal = lc_signal.signal[int(signal.n / 2):: signal.n]
        trans_signal['x_' + str(signal.Lnow)] = lc_signal
        # print(str(signal.Lnow) + 'kmまで伝送完了')
    save_pickle(trans_signal, 'dataset/trans_signal_tmp.pickle')
    return signal


def linear_compensation(signal):
    S = fft(signal.signal)
    S /= _H(signal, signal.Lnow)
    signal.signal = ifft(S)
    return signal


def _H(signal, L):  # H ファイバの伝達関数
    out = np.exp(-1j * signal.B2 * L / 2 * (signal.Omega - signal.Omega_c) ** 2)
    return out


def _sf(signal, x, y=None):
    if y is None:
        out = x * -1j * 2 * np.pi * signal.n2 / (2 * signal.Lambda * 10 ** -9) * abs(x ** 2)
    else:
        out = x * -1j * 8 / 9 * 2 * np.pi * signal.n2 \
              / (2 * signal.Lambda * 10 ** -9) * (abs(x) ** 2 + abs(y) ** 2)  # /2必要？
    return out


def _runge_kutta(signal, L, x, y=None):
    if y is None:
        k1 = _sf(signal, x) * L
        k2 = _sf(signal, x + k1 / 2) * L
        k3 = _sf(signal, x + k2 / 2) * L
        k4 = _sf(signal, x + k3) * L
        xx = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x * np.exp(-1j * L / 2 * 2 * np.pi * signal.n2
                       / (signal.Lambda * 10 ** -9) * (abs(x ** 2) + abs(xx ** 2)))
    else:
        k1 = _sf(signal, x, y) * L
        k2 = _sf(signal, x + k1 / 2, y) * L
        k3 = _sf(signal, x + k2 / 2, y) * L
        k4 = _sf(signal, x + k3, y) * L
        xx = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x * np.exp(-1j * L / 2 * 2 * np.pi * signal.n2
                       / (signal.Lambda * 10 ** -9) * (abs(x) ** 2 + abs(xx) ** 2))
    return x


def _add_ase_noise(signal, x):
    r = 2 * np.pi * np.random.rand(len(x))
    s = (signal.G - 1) * signal.nsp * signal.h * signal.f_c * signal.B
    ase_noise = np.sqrt((120 * np.pi / signal.nc) * s / (signal.Xi * signal.As)) * np.exp(1j * r)
    x += ase_noise
    return x


def _add_nonlinear_distortion(signal, x, y=None):
    dL = signal.Ledfa / signal.stepedfa  # dL[km] 区間一つの長さ
    dH = _H(signal, dL / 2)  # dH 区間一つの半分における伝達関数
    ddL = dL * 10 ** 3  # ddL[m]

    for i in range(signal.stepedfa):
        """1区間の前半の周波数領域での線形歪を計算"""
        X = fft(x)
        X = X * dH
        x = ifft(X)

        """伝搬損失の計算"""
        x = x * 10 ** (-signal.Alpha * dL / 2 / 10)  # Alpha = 0.20dB/km dL = 100km のとき1/100

        """1区間の実時間領域での非線形効果による位相推移をルンゲ・クッタ法を用いて計算"""
        x = _runge_kutta(signal, ddL, x, y)

        """1区間の後半の周波数領域での線形歪を計算"""
        X = fft(x)
        X = X * dH
        x = ifft(X)

        """伝搬損失の計算"""
        x = x * 10 ** (-signal.Alpha * dL / 2 / 10)  # Alpha = 0.20dB/km dL = 100km のとき1/100
    return x


if __name__ == '__main__':
    """
    QPSK信号のEVM検証
    熊本氏修士論文 P.42 Fig.4.12と比較を行った
    """
    PdBm_list = [i - 0.5 for i in range(-9, 12)]
    evm_result = np.zeros(len(PdBm_list), dtype=float)
    for i, PdBm in enumerate(PdBm_list):
        signal = Signal(type='Normal', form='RZ16QAM', N=11, n=32, boudrate=28, Lmax=1500, PdBm=PdBm, Ledfa=100,
                              stepedfa=30, gamma=1.5, D=16, Alpha=0.16, NF=3)
        transmission(signal)
        linear_compensation(signal)
        signal.input = signal.input[int(signal.n / 2):: signal.n]
        signal.signal = signal.signal[int(signal.n / 2):: signal.n]
        evm = signal.cal_evm_pr()
        print(evm)
        evm_result[i] = evm
