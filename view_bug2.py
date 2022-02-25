from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
import pickle
import os


def antispike(old_list_x, old_list_y):
    threshold = 0.02
    new_list_x = list()
    new_list_y = list()
    for index in range(1, len(old_list_x)):
        if abs(old_list_x[index] - old_list_x[index-1]) < threshold and abs(old_list_y[index] - old_list_y[index-1]) < threshold:
            new_list_x.append(old_list_x[index])
            new_list_y.append(old_list_y[index])
    return new_list_x, new_list_y


path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/results/')

stage = [mpimg.imread(path+'/media/stage_{}.png'.format(i)) for i in range(1, 5)]
color = {'BUG2': 'gold', 'PDDRL': 'dodgerblue', 'PDSRL': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink'}
sel = {'S1': 0, 'S2': 1, 'Su': 2, 'Sl': 3}

splitted_dir = list()
for dir in list_dir:
    if dir != 'data' and dir.split('_')[0] == 'BUG2':
        splitted_dir.append(dir.split('_'))
sorted_dir = sorted(splitted_dir, key=lambda row: row[1])
print('Dir:', sorted_dir)

for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    with open(path+'/results/'+'_'.join(directory)+'/BUG2', 'rb') as f:
        data = pickle.load(f)

    # values = [local_episode_list, episode_timing_list, episode_reward_list, position_list]
    name = directory[0]
    rewards = np.array(data[2])
    timings = np.array(data[1])
    x = list()
    y = list()
    for nx, ny in data[3]:
        x.append(nx)
        y.append(ny)

    sucess_list = list()
    for value in rewards:
        if value == 200:
            sucess_list.append(1)
        else:
            sucess_list.append(0)

    sucess_rate = (sum(sucess_list) / len(sucess_list)) * 100
    print('Data for', name, 'test simulations:')
    print('Sucess rate:', sucess_rate, "%")
    print('Episode reward mean:', rewards.mean())
    print('Episode reward std:', rewards.std())
    print('Episode timing mean:', timings.mean())
    print('Episode timing std:', timings.std())

    plt.imshow(stage[sel[directory[1]]], extent=[min(x) - 0.7, max(x) + 0.7, min(y) - 0.7, max(y) + 0.7])
    new_x = list()
    new_y = list()
    last = 0

    for i in range(len(x)-1):
        if abs(x[i]) > 0.5 and abs(x[i+1]) <= 0.5:
            new_x.append(x[last+1:i-1])
            new_y.append(y[last+1:i-1])
            last = i+1

    for x, y in zip(new_x, new_y):
        #x = x[5:-5]
        #y = y[5:-5]
        #x, y = antispike(x, y)
        x = np.array(x)
        y = np.array(y)
        x -= 0.2
        y += 0.15
        plt.plot(x, y, color='gold', linestyle='-', linewidth=1)

    plt.title('Path ' + name, size=20)
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    plt.show()
