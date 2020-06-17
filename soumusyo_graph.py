import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
https://www.soumu.go.jp/menu_news/s-news/01kiban04_02000160.html
https://www.soumu.go.jp/main_content/000671256.pdf
"""

n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
data = np.array([2889, 3560, 4448, 5467, 6840, 8232, 8027, 8903, 10289, 10976, 12086, 12650])

fig, ax = plt.subplots()
ax.plot(n, data, '-o')
plt.ylabel('Download traffic[Gbps]')
ax.set_xlim((-0.7, 10.7))
ax.set_ylim((1500, 14000))
ax.set_xticks(np.linspace(0, 12, 13))
ax.set_xticklabels(['2014-5', '2014-11', '2015-5', '2015-11', '2016-5', '2016-11', '2017-5', '2017-11', '2018-5', '2018-11', '2019-5', '2019-11'])
plt.gcf().autofmt_xdate()
plt.grid(which='major', axis='both', color='#999999', linestyle='--')
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
plt.show()