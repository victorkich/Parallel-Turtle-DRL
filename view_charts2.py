from matplotlib import pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

color = {'PDDRL-N': 'dodgerblue', 'PDSRL-N': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink',
         'DDPG-N': 'orange', 'DDPG-P': 'black', 'SAC-N': 'azure', 'SAC-P': 'brown'}
x_lim = {'S1': 100000, 'S2': 200000, 'Sl': 200000, 'Su': 200000, 'SL': 200000}
fig, ax = plt.subplots()

print('Generating charts...')
for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    directory = '_'.join(directory)+'.json'
    print(path+directory)
    with open(path+directory) as f:
        if any(directory == np.array(['DDPG-N', 'DDPG-P', 'SAC-N', 'SAC-P'])):
            data = json.load(f)
        else:
            data = json.load(f)

    rewards = data['y']
    rewards = np.array([200 if reward >= 200 else reward for reward in rewards])
    steps = data['x']

    means = lfilter(b, a, rewards)
    _, stds = mfilter(rewards, n)

    # sel = '-'.join([directory[0], directory[4]])
    sel = sorted_dir[c]
    print(sel)
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

########################################################################################################################
"""
path = os.path.dirname(os.path.abspath(__file__)) + '/saved_models/'
list_dir = os.listdir(path)
splitted_dir = list()
for dir in list_dir:
    splitted_dir.append(dir.split('_'))
sorted_dir = sorted(splitted_dir, key=lambda row: row[3])

n = 100  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {'PDDRL-N': 'dodgerblue', 'PDSRL-N': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink'}
x_lim = {'S1': 150, 'S2': 500, 'Sl': 2000, 'Su': 2000}
fig, ax = plt.subplots()

# sorted_dir = sorted_dir[-8:-4]
print('Dir:', sorted_dir)

print('Generating charts...')
for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    print(path + '_'.join(directory) + '/writer_data.json')
    with open(path + '_'.join(directory) + '/writer_data.json') as f:
        data = json.load(f)

    print('Directory:', directory)
    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]

    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    # print(data)
    rewards = pd.DataFrame(data['agent_0/reward']).iloc[:, 2].to_numpy()
    rewards = np.array([200 if reward >= 200 else reward for reward in rewards])
    # steps = pd.DataFrame(data['data_struct/global_step']).iloc[:, 2].to_numpy()
    episodes = np.arange(len(rewards))

    directory_name = np.array(directory)
    dumped = json.dumps({'x': episodes, 'y': rewards}, cls=NumpyEncoder)
    with open('_'.join(directory_name[[0, -2, -1]])+'.json', 'a') as f:
        f.write(dumped + '\n')

    #with open('_'.join(directory[[0, -2, -1]])+'.json', 'w') as outfile:
    #    json.dump({'x': episodes, 'y': rewards}, outfile)

    means = lfilter(b, a, rewards)
    _, stds = mfilter(rewards, n)

    sel = '-'.join([directory[0], directory[4]])
    ax.plot(episodes, means, linestyle='-', linewidth=2, label=sel if sel.split('-')[1] == 'P' else sel.split('-')[0],
            c=color[sel])
    ax.fill_between(episodes, means - stds, means + stds, alpha=0.15, facecolor=color[sel])

    if (c + 1) % 4 == 0:
        ax.legend(loc=4, prop={'size': 14})
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_xlim([0, x_lim[directory[3]]])
        ax.set_ylim([-21, 201])
        ax.grid()
        # plt.savefig("{}.pdf".format(sel), format="pdf", bbox_inches="tight", backend='pgf')
        plt.show()
        fig, ax = plt.subplots()

    del data
print('Done!')
"""
