from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import os

path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/real_results/')
threshold_x = 10
threshold_y = 30
threshold = 10
STAGE = 4
c = 7

def antispike(old_list_x, old_list_y):
    new_list_x = list()
    new_list_y = list()
    for index in range(1, len(old_list_x)):
        if abs(old_list_x[index] - old_list_x[index-1]) < threshold and abs(old_list_y[index] - old_list_y[index-1]) < threshold:
            new_list_x.append(old_list_x[index])
            new_list_y.append(old_list_y[index])
    return new_list_x, new_list_y

def antispike2(old_list_x, old_list_y):
    pivot_x = old_list_x[0]
    pivot_y = old_list_y[0]
    new_list_x = list()
    new_list_y = list()
    for index in range(1, len(old_list_x)):
        if abs(old_list_x[index] - pivot_x) < threshold and abs(old_list_y[index] - pivot_y) < threshold:
            pivot_x = old_list_x[index]
            pivot_y = old_list_y[index]
            new_list_x.append(old_list_x[index])
            new_list_y.append(old_list_y[index])
    return new_list_x, new_list_y


def open_test_data(i):
    return open(path + '/real_results/PDSRL_P_Sl_episode{}'.format(i), 'rb')


stage = mpimg.imread(path+'/media/stage_{}_real.png'.format(STAGE))

data = list()
for i in range(1, 15):
    with open_test_data(i) as f:
        data.append(pickle.load(f))

color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}

data = np.array(data)[[0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]]

size = len(data)
plt.imshow(stage)
rewards = list()
times = list()
for i in range(size):
    rewards.append(1 if data[i][0] == 20 else 0)
    times.append(data[i][1])

print(rewards)
print(times)
print('')
print('Valores testados:', size)
print('Time mean:', np.mean(times), 'std:', np.std(times))
print('Sucess rate:', (sum(rewards)/size) * 100, '%')

for l in range(size):
    x = []
    y = []
    for x_n, y_n in data[l][4]:
        x.append(x_n)
        y.append(y_n)

    x = np.array(x)
    x = x / 1.7
    x += 10
    y = np.array(y)
    y = y / 1.4
    y -= 10

    x, y = antispike(x, y)
    #x, y = antispike(x, y)

    plt.plot(x, y, color=color[c], linestyle='-', linewidth=2)
plt.show()
