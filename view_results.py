from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/results/')

stage1 = mpimg.imread(path+'/media/stage_1.png')
stage2 = mpimg.imread(path+'/media/stage_2.png')
# stage3 = mpimg.imread(path+'/media/stage_3.png')
# stage4 = mpimg.imread(path+'/media/stage_4.png')

plot = False
order = [True, False, True, True, False, True, False, False]
color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+'/results/'+directory+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]
    # print(new_key_list)

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

    print('Data for', directory, 'test simulations:')
    print('Episode reward mean:', df[new_key_list[0]].mean())
    print('Episode reward std:', df[new_key_list[0]].std())
    print('Episode timing mean:', df[new_key_list[1]].mean())
    print('Episode timing std:', df[new_key_list[1]].std())

    if plot:
        x = pd.DataFrame(data['agent_0/x']).iloc[:, 2].to_numpy().tolist()
        y = pd.DataFrame(data['agent_0/y']).iloc[:, 2].to_numpy().tolist()

        plt.imshow(stage2 if order[c] else stage1, extent=[min(x) - 0.7, max(x) + 0.7, min(y) - 0.7, max(y) + 0.7] if
                   order[c] else [min(x) - 0.95, max(x) + 0.95, min(y) - 0.95, max(y) + 0.95])

        new_x = list()
        new_y = list()
        last = 0
        for i in range(len(x)-1):
            if abs(x[i]) > 0.5 and abs(x[i+1]) <= 0.5:
                new_x.append(x[last+1:i-1])
                new_y.append(y[last+1:i-1])
                last = i+1

        for x, y in zip(new_x, new_y):
            plt.plot(y, x, color=color[c], linestyle='-', linewidth=1)

        plt.title('Path '+directory, size=20)
        plt.xlabel('Meters')
        plt.ylabel('Meters')
        plt.xlim([-2.1, 2.1])
        plt.ylim([-2.1, 2.1])
        plt.savefig("{}.pdf".format(directory), format="pdf", bbox_inches="tight")
        plt.show()
