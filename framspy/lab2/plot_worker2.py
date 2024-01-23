import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os
import pandas as pd
import matplotlib.lines as mlines

COLORS={
    '0': (1,0,0), 
    '1': (0,1,0), 
    '4': (0,0,1), 
    '9': (0.7,0,0.7), 
}

def color(gf, alpha=0.5):
    c = COLORS[gf][:3]
    c = c + tuple([alpha])
    return c

plt.figure(1)


for genformat in ['0', '1','4','9']:
    data2 = pd.DataFrame()
    for dir, line_type in zip(['logs2/prev/', 'logs2/'], ['-', '--']):
        for series in range(1,11):
            path = dir+f'logs_{genformat}_{series}.pickle'
            log = pickle.load(open(path, 'rb'))

            # preprocess data for 1st plot
            data1 = pd.DataFrame(log.select("nevals","max"), index=('nevals','max')).T
            for i in data1.index:
                if i==0: continue
                data1.loc[i, 'nevals'] = data1.loc[i, 'nevals'] + data1.loc[i-1, 'nevals']
            data1.set_index('nevals', inplace=True)

            plt.figure(1)
            plt.plot(data1, linewidth=0.5, c=color(genformat, 0.4), linestyle=line_type)

            # prepare data for 2nd plot
            data2_new_part = pd.DataFrame(
                log.select("gen","max"), index=('gen','max')
                ).T.set_index('gen')
            data2 = pd.concat([data2, data2_new_part], axis=1)
        
    
        # preprocess data for 2nd plot
        if line_type == '--':
            data2 = data2 #[:150]
        data2['avg'] = data2.mean(axis=1)
        data2['std'] = data2.std(axis=1)
        # print(data2)

        x = data2.index
        y = data2['avg']
        err = data2['std']/3
        plt.figure(2)
        plt.plot(x, y, c=color(genformat, 1), linestyle=line_type)
        plt.fill_between(x, y-err, y+err, color=color(genformat, 0.3))


handles = [mlines.Line2D([], [], color=COLORS[key], linestyle=line, label=key+lab) for key in COLORS.keys() for lab, line in zip([' before',' after'],['-','--'])]

plt.figure(1)
plt.xlabel('Liczba ocenionych osobników')
plt.ylabel('Maksymalna f. przystosowania')
plt.legend(handles=handles)
plt.savefig('lab2/1_max_fitnnes.png')

plt.figure(2)
plt.xlabel('Liczba ocenionych osobników ~(x50)')
plt.ylabel('Średnia f. przystosowania i jej std()')
plt.legend(handles=handles, loc='upper left')
plt.savefig('lab2/2_mean&std.png')

# PLOT 3
# fill dataframes
data3_fit = pd.DataFrame(
    index=range(1,11), columns=['0 before', '0 after', '1 before', '1 after', '4 before', '4 after', '9 before', '9 after'])
data3_time = pd.DataFrame(
    index=range(1,11), columns=['0 before', '0 after', '1 before', '1 after', '4 before', '4 after', '9 before', '9 after'])
for genformat in ['0','1','4','9']:
    for dir, line_type, label in zip(['HoFs2/prev/', 'HoFs2/'], ['-', '--'], [' before',' after']):
        for series in range(1,11):
            file_name = f'HoF-f{genformat}-{series}.gen'
            with open(dir+file_name, 'r') as f:
                HoF = f.read().split('\n')
                f_val = float(next((row for row in HoF if row.startswith("vertpos:")), 'vertpos:0')[8:])
                time = float(HoF[-1][2:]) 
                data3_fit.at[series, genformat+label] = f_val
                data3_time.at[series, genformat+label] = time

# labels = ['0','1','4','9']
labels = data3_fit.columns
print(labels)
plt.figure(3)
plt.ylabel('Średnia wartość f. z HoF')
plt.boxplot(data3_fit, labels=labels)
plt.savefig('lab2/3.1_mean_HoF.png')
plt.figure(4)
plt.ylabel('Czas [s]')
plt.boxplot(data3_time, labels=labels)
plt.savefig('lab2/3.2_Czas.png')

plt.show()