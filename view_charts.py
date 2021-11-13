from matplotlib import pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/saved_models/'
list_dir = os.listdir(path)

n = 50  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+directory+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]
    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    rewards = pd.DataFrame(data[new_key_list[0]]).iloc[:, 2].to_numpy()
    episodes = np.arange(len(rewards))
    mean = lfilter(b, a, rewards)

    plt.plot(episodes, mean, color=color[c], linestyle='-', linewidth=2, label=directory)
    plt.plot(episodes, rewards, color=color[c], linestyle='-', linewidth=1, alpha=0.3)  # label='Real Rewards'

plt.title('Reward per Episode', size=20)
plt.legend(loc=0, prop={'size': 12})
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.xlim([0, 1000])
plt.ylim([0, 200])
plt.show()
