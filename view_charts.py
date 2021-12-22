from matplotlib import pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm
import seaborn as sns
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
list_dir = sorted(os.listdir(path))
l1 = list()
l2 = list()
for i in range(len(list_dir)):
    if not i % 2 == 0:
        l1.append(list_dir[i])
    else:
        l2.append(list_dir[i])
list_dir = l1 + l2
list_dir = [list_dir[0], list_dir[2], list_dir[4], list_dir[6], list_dir[1], list_dir[3], list_dir[5], list_dir[7]]

n = 50  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
label = ['PDDRL-P', 'PDSRL-P', 'PDDRL', 'PDSRL', 'PDDRL-P', 'PDSRL-P', 'PDDRL', 'PDSRL']
clrs = sns.color_palette("husl", len(list_dir))
fig, ax = plt.subplots()

for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+directory+'/writer_data.json') as f:
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

    ax.plot(episodes, means, linestyle='-', linewidth=2, label=label[c], c=color[c])
    ax.fill_between(episodes, means - stds, means + stds, alpha=0.15, facecolor=color[c])

    if (c+1) % 4 == 0:
        ax.legend(loc=4, prop={'size': 14})
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_xlim([0, 800])
        ax.set_ylim([-21, 201])
        ax.grid()
        plt.show()
        fig, ax = plt.subplots()

"""
from matplotlib import pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/saved_models/'
list_dir = sorted(os.listdir(path))
l1 = list()
l2 = list()
for i in range(len(list_dir)):
    if not i % 2 == 0:  
        l1.append(list_dir[i])
    else:
        l2.append(list_dir[i])
list_dir = l1 + l2
list_dir = [list_dir[0], list_dir[2], list_dir[4], list_dir[6], list_dir[1], list_dir[3], list_dir[5], list_dir[7]]

n = 50  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
label = ['PDDRL-P', 'PDSRL-P', 'PDDRL', 'PDSRL', 'PDDRL-P', 'PDSRL-P', 'PDDRL', 'PDSRL']
clrs = sns.color_palette("husl", len(list_dir))
fig, ax = plt.subplots()

for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+directory+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]
    # print(new_key_list)
    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    rewards = pd.DataFrame(data['agent_0/reward']).iloc[:, 2].to_numpy()
    rewards = np.array([200 if reward >= 200 else reward for reward in rewards])
    episodes = np.arange(len(rewards))
    steps = pd.DataFrame(data['data_struct/global_step']).iloc[:, 2].to_numpy()
    means = lfilter(b, a, rewards)
    stds = list()
    dual = False
    for i in range(len(means)):
        if dual:
            stds.append(np.std(means[i:], ddof=1) + np.std(rewards[i:], ddof=1))
        else:
            stds.append(np.std(rewards[i:], ddof=1))
    # stds = lfilter(b, a, stds)
    # stds = np.std(rewards, ddof=1)

    ax.plot(episodes, means, linestyle='-', linewidth=2, label=label[c], c=color[c])
    ax.fill_between(episodes, means - stds, means + stds, alpha=0.25, facecolor=color[c])

    if (c+1) % 4 == 0:
        ax.legend(loc=4, prop={'size': 14})
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_xlim([0, 800])
        ax.set_ylim([-21, 201])
        ax.grid()
        plt.show()
        fig, ax = plt.subplots()

"""
