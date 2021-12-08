from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import json
import os

path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + '/results/')

stage1 = mpimg.imread(path+'/media/stage_1.png')
stage2 = mpimg.imread(path+'/media/stage_2.png')

order = [True, False, True, True, False, True, False, False]
color = {0: 'firebrick', 1: 'tomato', 2: 'peru', 3: 'gold', 4: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 7: 'deeppink'}
for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+'/results/'+directory+'/writer_data.json') as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split('/')[-2:]) for key in key_list]

    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

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
    # plt.legend(loc=4, prop={'size': 12})
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    plt.xlim([-2.1, 2.1])
    plt.ylim([-2.1, 2.1])
    plt.show()
