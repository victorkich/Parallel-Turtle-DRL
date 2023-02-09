from matplotlib import pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os


def mfilter(array, size):
    numbers = array
    window_size = size

    i = 0 
    moving_averages = []
    std_array = []
    while i < len(numbers): 
        this_window = numbers[i-int(window_size/2) if i - int(window_size/2) >= 0 else 0: i + int(window_size/2)+1]
        std_array.append(np.std(this_window)) 
        window_average = sum(this_window) / (window_size if len(this_window) == window_size else len(this_window)) 
        moving_averages.append(window_average) 
        i += 1
    return np.array(moving_averages), np.array(std_array)


path = os.path.dirname(os.path.abspath(__file__)) + '/classics/'
list_dir = os.listdir(path)
splitted_dir = list()
for dir in list_dir:
    splitted_dir.append(dir[:-5].split('_'))
sorted_dir = sorted(splitted_dir, key=lambda row: row[2])
print(sorted_dir)

n = 100  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {'DDPG-N': 'dodgerblue', 'DDPG-P': 'springgreen', 'SAC-N': 'indigo', 'SAC-P': 'deeppink'}
x_lim = {'S1': 100000, 'S2': 200000, 'Sl': 200000, 'Su': 200000, 'SL': 200000}
fig, ax = plt.subplots()

print('Generating charts...')
for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    directory = '_'.join(directory)+'.json'
    print(path+directory)
    with open(path+directory) as f:
        data = json.load(f)[0]

    rewards = data['y']
    rewards = np.array([200 if reward >= 200 else reward for reward in rewards])
    steps = data['x']

    means = lfilter(b, a, rewards)
    _, stds = mfilter(rewards, n)

    # sel = '-'.join([directory[0], directory[4]])
    sel = sorted_dir[c]
    ax.plot(steps, means, linestyle='-', linewidth=2, label='-'.join(sel[:-1]), c=color['-'.join(sel[:-1])])
    ax.fill_between(steps, means - stds, means + stds, alpha=0.15, facecolor=color['-'.join(sel[:-1])])

    if (c+1) % 4 == 0:
        ax.legend(loc=4, prop={'size': 14})
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_xlim([0, x_lim[sel[-1]]])
        ax.set_ylim([-21, 201])
        ax.grid()
        # print("Saving at:", "{}.pdf".format(sel))
        # plt.savefig("{}.pdf".format(sel), format="pdf", bbox_inches="tight", backend='pgf')
        plt.show()
        fig, ax = plt.subplots()

    del data
print('Done!')
