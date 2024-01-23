import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os
import pandas as pd
import matplotlib.lines as mlines

COLORS={
    # '0': (1,0,0), 
    # '1': (0,1,0), 
    # '2': (0,0,1), 
    # '3': (0.7,0,0.7), 
    # '4': (0.6,0.6,0),
    'GP': (1,0,0),
    'native': (0,1,0)
}

def color(gf, alpha=0.5):
    c = COLORS[gf][:3]
    c = c + tuple([alpha])
    return c

SERIES = [i for i in range(1,13)]
MODES = ["GP", "native"]

fig, axs = plt.subplots(2,2)
plt.figure(1)

for mode in MODES:
    data2_mean = pd.DataFrame()
    for series in SERIES:
        path = f'output/{mode}/log_{series}.pickle'
        log = pickle.load(open(path, 'rb'))

        # preprocess data for 1st plot
        data1_max = pd.DataFrame(log.select("nevals","max"), index=('nevals','max')).T
        for i in data1_max.index:
            if i==0: continue
            data1_max.loc[i, 'nevals'] = data1_max.loc[i, 'nevals'] + data1_max.loc[i-1, 'nevals']
        data1_max.set_index('nevals', inplace=True)

        # plt.figure(1)
        axs[0,0].plot(data1_max, linewidth=0.5, c=color(mode, 0.4), linestyle='-')

        # prepare data for 2nd plot
        data2_new_part = pd.DataFrame(
            log.select("gen","max"), index=('gen','max')
            ).T.set_index('gen')
        data2_mean = pd.concat([data2_mean, data2_new_part], axis=1)

    # preprocess data for 2nd plot
    data2_mean['avg'] = data2_mean.mean(axis=1)
    data2_mean['std'] = data2_mean.std(axis=1)

    x = data2_mean.index
    y = data2_mean['avg']
    err = data2_mean['std']/3
    # plt.figure(2)
    axs[0,1].plot(x, y, c=color(mode, 1), linestyle='-')
    axs[0,1].fill_between(x, y-err, y+err, color=color(mode, 0.3))


handles = [
    mlines.Line2D([], [], color=COLORS[key], linestyle='-', label=key) 
    for key in COLORS.keys()
]

axs[0,0].set_xlabel('Liczba ocenionych osobników')
axs[0,0].set_ylabel('Maksymalna f. przystosowania')
axs[0,0].legend(handles=handles)
# axs[0,0].savefig('1_max_fitnnes.png')

# plt.figure(2)
axs[0,1].set_xlabel('Liczba ocenionych osobników ~(x50)')
axs[0,1].set_ylabel('Średnia f. przystosowania i jej std()')
axs[0,1].legend(handles=handles, loc='upper left')
# axs[0,1].savefig('2_mean&std.png')

# PLOT 3
# fill dataframes
data3_fit = pd.DataFrame(index=SERIES, columns=MODES)
data4_time = pd.DataFrame(index=SERIES, columns=MODES)
for mode in MODES:
    for series in SERIES:
        with open(f'output/{mode}/HoF_{series}.gen', 'r') as f:
            HoF = f.read().split('\n')
            f_val = float(next((row for row in HoF if row.startswith("vertpos:")), 'vertpos:0')[9:])
            data3_fit.at[series, mode] = f_val
        with open(f'output/{mode}/time_{series}.txt', 'r') as f:
            time = f.readline()
            data4_time.at[series, mode] = float(time)

labels = data3_fit.columns
print(labels)
# plt.figure(3)
axs[1,0].set_ylabel('Średnia wartość f. z HoF')
axs[1,0].boxplot(data3_fit, labels=labels)
# plt.ylim(-0.001, 0.015)
# plt.savefig('3.1_mean_HoF.png')

# plt.figure(4)
axs[1,1].set_ylabel('Czas [s]')
axs[1,1].boxplot(data4_time, labels=labels)
# plt.savefig('3.2_Czas.png')
plt.savefig('plots.png')

plt.show()