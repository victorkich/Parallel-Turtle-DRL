from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/results/')

stage = [mpimg.imread(path+'/media/stage_{}.png'.format(i)) for i in range(1, 5)]
color = {'BUG2': 'gold', 'PDDRL': 'dodgerblue', 'PDSRL': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink'}
sel = {'S1': 0, 'S2': 1, 'Su': 2, 'Sl': 3}

splitted_dir = list()
for dir in list_dir:
    if dir != 'data' and dir.split('_')[0] != 'BUG2':
        splitted_dir.append(dir.split('_'))
sorted_dir = sorted(splitted_dir, key=lambda row: row[1] if row[0] == 'BUG2' else row[3])
print('Dir:', sorted_dir)

sorted_dir = sorted_dir[14:]

for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    with open(path+'/results/'+'_'.join(directory)+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]

    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    df = pd.DataFrame(data, dtype=np.float32)
    reward = df.iloc[:, df.columns == new_key_list[0]].to_numpy()
    new_reward = list()
    for i, t in enumerate(reward):
        new_reward.append(t[0][-1])

    timing = df.iloc[:, df.columns == new_key_list[1]].to_numpy()
    new_timing = list()
    for i, t in enumerate(timing):
        new_timing.append(t[0][-1])

    episode = df.iloc[:, df.columns == new_key_list[2]].to_numpy()
    new_episode = list()
    for i, t in enumerate(episode):
        new_episode.append(t[0][-1])

    df = pd.DataFrame({new_key_list[0]: list(new_reward), new_key_list[1]: list(new_timing), new_key_list[2]: list(new_episode)}, dtype=np.float32)
    df = df.sort_values([new_key_list[2], new_key_list[0], new_key_list[1]], ascending=[True, False, False])
    df = df.groupby(new_key_list[2]).first().reset_index()[1:]

    if directory[0] != 'BUG2':
        if directory[-1] != 'N':
            name = "-".join([directory[0], directory[-1]])
        else:
            name = directory[0]
        c = directory[-2]
    else:
        name = directory[0]
        c = directory[1]

    sucess_list = list()
    for value in df[new_key_list[0]]:
        if value == 200:
            sucess_list.append(1)
        else:
            sucess_list.append(0)

    sucess_rate = (sum(sucess_list) / len(sucess_list)) * 100
    print('Data for', name, 'test simulations:')
    print('Sucess rate:', sucess_rate, "%")
    print('Episode reward mean:', df[new_key_list[0]].mean())
    print('Episode reward std:', df[new_key_list[0]].std())
    print('Episode timing mean:', df[new_key_list[1]].mean())
    print('Episode timing std:', df[new_key_list[1]].std())

    x = pd.DataFrame(data['agent_0/x']).iloc[:, 2].to_numpy().tolist()
    y = pd.DataFrame(data['agent_0/y']).iloc[:, 2].to_numpy().tolist()

    plt.imshow(stage[sel[c]], extent=[min(x) - 0.7, max(x) + 0.7, min(y) - 0.7, max(y) + 0.7])

    new_x = list()
    new_y = list()
    last = 0
    for i in range(len(x)-1):
        if abs(x[i]) > 0.5 and abs(x[i+1]) <= 0.5:
            new_x.append(x[last+1:i-1])
            new_y.append(y[last+1:i-1])
            last = i+1

    for x, y in zip(new_x, new_y):
        x = np.array(x)
        y = np.array(y)
        x -= 0.3
        y += 0.2
        plt.plot(x, y, color=color[name], linestyle='-', linewidth=1)

    plt.title('Path ' + name, size=20)
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    plt.show()
