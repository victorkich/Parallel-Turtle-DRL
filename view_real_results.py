from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
import pickle
import os


path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/saved_models/')
threshold = 10

color = {'PDDRL-N': 'dodgerblue', 'PDSRL-N': 'springgreen', 'PDDRL-P': 'indigo', 'PDSRL-P': 'deeppink',
         'DDPG-N': 'orange', 'DDPG-P': 'black', 'SAC-N': 'darkslategray', 'SAC-P': 'brown', 'BUG2-N': 'gold'}

splitted_dir = list()
for dir in list_dir:
    splitted_dir.append(dir.split('_'))

sorted_dir = sorted(splitted_dir, key=lambda row: row[4])
sorted_dir = sorted(sorted_dir, key=lambda row: row[0])
sorted_dir = sorted(sorted_dir, key=lambda row: row[4])
sorted_dir = sorted(sorted_dir, key=lambda row: row[3])

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
fig.set_dpi(100)

# print('Sorted dir:', sorted_dir)
# sorted_dir = sorted_dir[33:]

for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    directory = [directory[0], directory[4], directory[3]]
    file_name = '_'.join(directory[:3])
    data = list()
    try:
        for i in range(0, 12):
            with open(path + '/real_results_extra/{}_episode{}.pkl'.format(file_name, i), 'rb') as f:
                data.append(pickle.load(f))
    except:
        print('C:', c)
        continue

    stage = mpimg.imread(path + '/media/stage_extra_real.png'.format(directory[2]))
    ax.imshow(stage)

    size = len(data)
    rewards = list()
    times = list()
    for i in range(size):
        rewards.append(1 if data[i][0] == 20 else 0)
        times.append(round(data[i][1], 2))

    print('Algorithm:', file_name)
    print('Rewards:', rewards)
    print('Times:', times)
    print('Valores testados:', size)
    print('Time mean:', round(np.mean(times), 2), 'std:', round(np.std(times), 2))
    print('Sucess rate:', (sum(rewards) / size) * 100, '%')

    for l in range(size):
        x = []
        y = []
        for x_n, y_n in data[l][4]:
            x.append(x_n)
            y.append(y_n)

        x = np.array(x)
        y = np.array(y)

        x = x * 1.25
        x += 10
        y = y * 1.25
        y += 10

        ax.plot(x, y, color=color['-'.join(directory[:-1])], linestyle='-', linewidth=2)

    plt.title('Path ' + file_name, size=20)
    ax.set_xlabel('Step', fontsize=20)
    ax.set_ylabel('Reward', fontsize=20)
    plt.show()
    plt.show()

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    fig.set_dpi(100)
