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


path = os.path.dirname(os.path.abspath(__file__)) + '/saved_models/'
list_dir = os.listdir(path)
splitted_dir = list()
for dir in list_dir:
    splitted_dir.append(dir.split('_'))
sorted_dir = sorted(splitted_dir, key=lambda row: row[3])

n = 50  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {'PDDRL-N': 'dodgerblue', 'PDSRL-N': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink'}
x_lim = {'S1': 150, 'S2': 1000, 'Sl': 2000, 'Su': 2000}
fig, ax = plt.subplots()

print('Generating charts...')
for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    with open(path+'_'.join(directory)+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]

    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    rewards = pd.DataFrame(data['agent_0/reward']).iloc[:, 2].to_numpy()
    rewards = np.array([200 if reward >= 200 else reward for reward in rewards])
    episodes = np.arange(len(rewards))
    steps = pd.DataFrame(data['data_struct/global_step']).iloc[:, 2].to_numpy()
    means = lfilter(b, a, rewards)
    _, stds = mfilter(rewards, n)

    sel = '-'.join([directory[0], directory[4]])
    print(sel)
    ax.plot(episodes, means, linestyle='-', linewidth=2, label=sel, c=color[sel])
    ax.fill_between(episodes, means - stds, means + stds, alpha=0.15, facecolor=color[sel])

    if (c+1) % 4 == 0:
        ax.legend(loc=4, prop={'size': 14})
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_xlim([0, x_lim[directory[3]]])
        ax.set_ylim([-21, 201])
        ax.grid()
        plt.savefig("{}.pdf".format(x_lim[directory[3]]), format="pdf", bbox_inches="tight")
        plt.show()
        fig, ax = plt.subplots()

print('Done!')
