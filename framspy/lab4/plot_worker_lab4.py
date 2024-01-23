import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os
import pandas as pd
import matplotlib.lines as mlines

COLORS={
    '0': (1,0,0), 
    '1': (0,1,0), 
    '2': (0,0,1), 
    '3': (0.7,0,0.7), 
    '4': (0.6,0.6,0)
}

def color(gf, alpha=0.5):
    c = COLORS[gf][:3]
    c = c + tuple([alpha])
    return c

plt.figure(1)

gen_formats = ['0', '1', '2', '3', '4'] #3
for genformat in gen_formats:
    data2 = pd.DataFrame()
    for series in range(1,11):
        path = f'logs/logs_{genformat}-{series}.pickle'
        log = pickle.load(open(path, 'rb'))

        # preprocess data for 1st plot
        data1 = pd.DataFrame(log.select("nevals","max"), index=('nevals','max')).T
        for i in data1.index:
            if i==0: continue
            data1.loc[i, 'nevals'] = data1.loc[i, 'nevals'] + data1.loc[i-1, 'nevals']
        data1.set_index('nevals', inplace=True)

        plt.figure(1)
        plt.plot(data1, linewidth=0.5, c=color(genformat, 0.4), linestyle='-')

        # prepare data for 2nd plot
        data2_new_part = pd.DataFrame(
            log.select("gen","max"), index=('gen','max')
            ).T.set_index('gen')
        data2 = pd.concat([data2, data2_new_part], axis=1)
    

    # preprocess data for 2nd plot
    data2['avg'] = data2.mean(axis=1)
    data2['std'] = data2.std(axis=1)
    # print(data2)

    x = data2.index
    y = data2['avg']
    err = data2['std']/3
    plt.figure(2)
    plt.plot(x, y, c=color(genformat, 1), linestyle='-')
    plt.fill_between(x, y-err, y+err, color=color(genformat, 0.3))


handles = [
    mlines.Line2D([], [], color=COLORS[key], linestyle='-', label=key) 
    for key in COLORS.keys()
]

plt.figure(1)
plt.xlabel('Liczba ocenionych osobników')
plt.ylabel('Maksymalna f. przystosowania')
plt.legend(handles=handles)
plt.savefig('1_max_fitnnes.png')

plt.figure(2)
plt.xlabel('Liczba ocenionych osobników ~(x50)')
plt.ylabel('Średnia f. przystosowania i jej std()')
plt.legend(handles=handles, loc='upper left')
plt.savefig('2_mean&std.png')

# PLOT 3
# fill dataframes
data3_fit = pd.DataFrame(
    index=range(1,11), columns=gen_formats)
data3_time = pd.DataFrame(
    index=range(1,11), columns=gen_formats)
for genformat in gen_formats:
    for series in range(1,11):
        file_path = f'HoFs/HoF-f{genformat}-{series}.gen'
        with open(file_path, 'r') as f:
            HoF = f.read().split('\n')
            f_val = float(next((row for row in HoF if row.startswith("velocity:")), 'velocity:0')[9:])
            time = float(HoF[-1][2:]) 
            data3_fit.at[series, genformat] = f_val
            data3_time.at[series, genformat] = time

# labels = ['0','1','4','9']
labels = data3_fit.columns
print(labels)
plt.figure(3)
plt.ylabel('Średnia wartość f. z HoF')
plt.boxplot(data3_fit, labels=labels)
# plt.ylim(-0.001, 0.015)
plt.savefig('3.1_mean_HoF.png')
plt.figure(4)
plt.ylabel('Czas [s]')
plt.boxplot(data3_time, labels=labels)
plt.savefig('3.2_Czas.png')

plt.show()