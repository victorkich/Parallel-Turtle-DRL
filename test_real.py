#! /usr/bin/env python3

import rospy
from utils import range_finder as rf
import gym_turtlebot3
from models import PolicyNetwork, TanhGaussianPolicy
from utils.defisheye import Defisheye
from algorithms.bug2 import BUG2
import multiprocessing as mp
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from math import isnan
from cv_bridge import CvBridge
import pandas as pd
import numpy as np
import imutils
import torch
import yaml
import time
import gym
import cv2
import os

TURTLE = '004'
bridge = CvBridge()
state = None
font = cv2.FONT_HERSHEY_SIMPLEX

# Hyper parameters
episodes = 10
max_steps = 1000
action_low = [-1.5, -0.1]
action_high = [1.5, 0.12]

# Loading configs from config.yaml
path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/config.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

env = input('Which environment are you running? [1 | 2 | l | u]:\n')
rospy.init_node(config['env_name'].replace('-', '_') + "_test_real")
env_real = gym.make(config['env_name'], env_stage=env.lower(), observation_mode=0, continuous=True, test_real=True)
_ = env_real.reset()
real_ttb = rf.RealTtb(config, path, output=(640, 640))
defisheye = Defisheye(dtype='linear', format='fullframe', fov=100, pfov=90)


def getImage(image):
    global state
    start = time.time()
    try:
        lidar = rospy.wait_for_message('scan_' + TURTLE, LaserScan, timeout=3)
    except:
        pass
    frame = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    frame = imutils.rotate_bound(frame, 2)
    frame = defisheye.convert(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    angle = distance = None
    try:
        lidar = np.array(lidar.ranges)
        lidar = np.array([max(lidar[[i - 1, i, i + 1]]) for i in range(7, 361, 15)]).squeeze()
        angle, distance, frame = real_ttb.get_angle_distance(frame, lidar, green_magnitude=1.0)
    except:
        pass

    if not angle is None and not distance is None:
        state = np.hstack([lidar, angle, distance])

    fps = round(1 / (time.time() - start))
    # putting the FPS count on the frame
    cv2.putText(frame, 'FPS: ' + str(fps), (7, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass


sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)

path_results = path + '/real_results'
if not os.path.exists(path_results):
    os.makedirs(path_results)

if not os.path.exists(path_results+'/real_results_S{}.csv'.format(env)):
    print('File real_results_S{}.csv not found!\nGeneraring a new file real_results_S{}.csv ...'.format(env, env))
    df = pd.DataFrame({'PDDRL': [], 'PDSRL': [], 'PDDRL-P': [], 'PDSRL-P': [], 'DDPG': [], 'SAC': [], 'BUG2': []})
    df.to_csv(path_results+'/real_results_S{}.csv'.format(env))
else:
    print('File real_results_S{}.csv found!\nLoading data from real_results_S{}.csv...'.format(env, env))
    df = pd.read_csv(path_results+'/real_results_S{}.csv'.format(env))
data = df.to_dict()

time.sleep(1)
translator = {1: ['PDDRL', 'N'], 2: ['PDSRL', 'N'], 3: ['PDDRL', 'P'], 4: ['PDSRL', 'P'], 5: ['DDPG', 'N'],
              6: ['SAC', 'N'], 7: ['BUG2', 'N']}
algorithms_sel = np.array(['1', '2', '3', '4', '5', '6', '7', 'e', 'r'])
algorithm = ""
while True:
    while not any(algorithm.lower() == algorithms_sel):
        print('Choose the algorithm or exit the test:')
        algorithm = input('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->BUG2 | '
                          'e->exit | r->reset\n')
    if algorithm.lower() == 's':
        break
    if algorithm.lower() == 'r':
        reset = input('Do you want to reset any test results? [y/n]\n')
        if reset.lower() == 'y':
            reset = input('Do you want to reset all the test results? [y/n]\n')
            if reset.lower() == 'y':
                df = pd.DataFrame({'PDDRL': [], 'PDSRL': [], 'PDDRL-P': [], 'PDSRL-P': [], 'DDPG': [], 'SAC': [], 'BUG2': []})
                df.to_csv(path_results + '/real_results.csv')
            else:
                column = None
                while not any(column == np.arange(7)+1):
                    print('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->BUG2')
                    column = input('Which columns do you want reset?\n')
                data[list(data.keys())[int(column) - 1]] = []
                df = pd.DataFrame.from_dict(data, orient='index').T
                df.to_csv(path_results + '/real_results_S{}.csv'.format(env))
        else:
            continue

    if algorithm != '7':
        process_dir = f"{path}/saved_models/{translator[int(algorithm)][0]}_{config['dense_size']}_A{config['num_agents']}_S{env}_{'P' if config['replay_memory_prioritized'] else 'N'}"
        list_dir = sorted(os.listdir(process_dir))
        model_fn = f"{process_dir}/{list_dir[-2]}"

        # Loading neural network model
        if any(algorithm == algorithms_sel[[0, 2]]):
            actor = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], device=config['device'])
        elif any(algorithm == algorithms_sel[[1, 3]]):
            actor = TanhGaussianPolicy(config=config, obs_dim=config['state_dim'], action_dim=config['action_dim'],
                                       hidden_sizes=[config['dense_size'], config['dense_size']])
        try:
            actor.load_state_dict(torch.load(model_fn, map_location=config['device']))
        except:
            actor = torch.load(model_fn)
            actor.to(config['device'])
        actor.eval()
    else:
        b2 = BUG2()

    local_episode = len(data[list(data.keys())[int(algorithm) - 1]])
    while local_episode <= episodes:
        if local_episode == 0:
            quit = input("Press [Enter] to start the test or press [q] to quit...")
        else:
            quit = input("Press [Enter] to continue to the next episode or press [q] to quit...")
        if quit.lower() == 'q':
            break

        episode_reward = 0
        xy = list()
        lidar_list = list()
        num_steps = 0
        local_episode += 1
        ep_start_time = time.time()
        done = False
        while True:
            start = time.time()
            print('Num steps:', num_steps)
            print('State:', state)
            if algorithm != '7':
                action = actor.get_action(torch.Tensor(state).to(config['device']) if algorithm == '2' or algorithm == '4' else np.array(state))
                action = action.detach().cpu().numpy().flatten()
                action[0] = np.clip(action[0], action_low[0], action_high[0])
                action[1] = np.clip(action[1], action_low[1], action_high[1])
            else:
                action = b2.get_action(state)

            print('Action:', action)
            _, _, _, _ = env_real.step(action=action)
            reward, done = env_real.get_done_reward(lidar=state[0:24], distance=state[-1])
            episode_reward += reward

            position = env_real.get_position()  # Get x and y turtlebot position to compute test charts
            scan = env_real.get_scan()
            xy.append(position)
            lidar_list.append(scan)

            print('Done:', done)
            if done or num_steps == max_steps:
                cv2.destroyAllWindows()
                break
            else:
                num_steps += 1

            print('Step timing:', time.time() - start)
            fps = round(1 / (time.time() - start))
            print('FPS:', fps)

        # Log metrics
        episode_timing = time.time() - ep_start_time
        print(f"Agent: [Test] Episode: [{local_episode}/{episodes}] Reward: [{episode_reward}/200] "
              f"Steps: [{num_steps}/{max_steps}] Episode Timing: {round(episode_timing, 2)}s")

        # Save csv file
        values = [episode_reward, episode_timing, local_episode, num_steps, xy, lidar_list]
        data[list(data.keys())[int(algorithm) - 1]] = list(filter(lambda k: not isnan(k), data[list(data.keys())[int(algorithm) - 1]]))
        data[list(data.keys())[int(algorithm) - 1]].append(values)
        df = pd.DataFrame.from_dict(data, orient='index').T
        df.to_csv(path_results + '/real_results_S{}.csv'.format(env))
    print('Done!')
p.join()
