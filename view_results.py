from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/results/'
list_dir = os.listdir(path)

n = 75  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
# for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
with open(path+list_dir[0]+'/writer_data.json') as f:
    data = json.load(f)

key_list = list(data.keys())
new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]
print(new_key_list)
for i, key in enumerate(key_list):
    data[new_key_list[i]] = data.pop(key)

x = -pd.DataFrame(data['agent_0/x']).iloc[:, 2].to_numpy()
x = x.tolist()
y = pd.DataFrame(data['agent_0/y']).iloc[:, 2].to_numpy().tolist()

new_x = []
new_y = []
last = 0
for i in range(len(x)-1):
    if abs(x[i]) > 1 and abs(x[i+1]) <= 0.2:
        new_x.append(x[last:i+1])
        new_y.append(y[last:i+1])
        last = i+1

for x, y in zip(new_x, new_y):
    plt.plot(x, y, color=color[0], linestyle='-', linewidth=1)  # label=list_dir[0]
    # plt.plot(episodes, rewards, color=color[0], linestyle='-', linewidth=1, alpha=0.25)

plt.title('Path '+list_dir[0], size=20)
plt.legend(loc=4, prop={'size': 12})
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
