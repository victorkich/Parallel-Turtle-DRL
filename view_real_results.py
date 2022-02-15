from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json
import os

path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/real_results/')


def open_test_data(i):
    return open(path + '/real_results/PDSRL_P_S2_episode{}'.format(i), 'rb')


#stage1 = mpimg.imread(path+'/media/stage_2_real.png')

data = list()
for i in range(1, 13):
    with open_test_data(i) as f:
        data.append(pickle.load(f))

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}

c = 7
#plt.imshow(stage1)
rewards = list()
times = list()
for i in range(12):
    rewards.append(1 if data[i][0] == 20 else 0)
    times.append(data[i][1])

print(rewards)
print(times)
print('')
print('Time mean:', np.mean(times), 'std:', np.std(times))
print('Sucess rate:', (sum(rewards)/12) * 100, '%')

#for l in range(12):

#    pass
    #x = []
    #y = []
    #for x_n, y_n in data[l][4]:
    #    x.append(x_n)
    #    y.append(y_n)

    #plt.plot(x, y, color=color[c], linestyle='-', linewidth=2)
#plt.show()

# stage3 = mpimg.imread(path+'/media/stage_3.png')
# stage4 = mpimg.imread(path+'/media/stage_4.png')

