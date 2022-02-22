#! /usr/bin/env python3

import rospy
from utils import range_finder as rf
import gym_turtlebot3
from models import PolicyNetwork, TanhGaussianPolicy
from utils.defisheye import Defisheye
from algorithms.bug2 import BUG2
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from tempfile import TemporaryFile
from cv_bridge import CvBridge
import numpy as np
import pickle
import imutils
import torch
import yaml
import time
import gym
import cv2
import os

TURTLE = '003'
bridge = CvBridge()
state = None
font = cv2.FONT_HERSHEY_SIMPLEX
outfile = TemporaryFile()

# Hyper parameters
episodes = 50
max_steps = 500
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
    try:
        lidar = rospy.wait_for_message('scan_' + TURTLE, LaserScan, timeout=1)
    except:
        pass
    frame = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    frame = imutils.rotate_bound(frame, 2)
    frame = defisheye.convert(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    angle = distance = None
    try:
        lidar = np.array(lidar.ranges)
        lidar = np.array([min(lidar[[i - 1, i, i + 1]]) for i in range(7, 361, 15)]).squeeze()
        angle, distance, frame = real_ttb.get_angle_distance(frame, lidar, green_magnitude=1.0)
        distance += 0.10
    except:
        pass

    if not angle is None and not distance is None:
        state = np.hstack([lidar, angle, distance])

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass


sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)

path_results = path + '/real_results'
if not os.path.exists(path_results):
    os.makedirs(path_results)

time.sleep(1)
translator = {1: ['PDDRL', 'N'], 2: ['PDSRL', 'N'], 3: ['PDDRL', 'P'], 4: ['PDSRL', 'P'], 5: ['DDPG', 'N'],
              6: ['SAC', 'N'], 7: ['BUG2', 'N']}
algorithms_sel = np.array(['1', '2', '3', '4', '5', '6', '7', 'e', 'r'])
while True:
    algorithm = ""
    while not any(algorithm.lower() == algorithms_sel):
        print('Choose the algorithm or exit the test:')
        algorithm = input('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->BUG2 | '
                          'e->exit | r->reset\n')
    if algorithm.lower() == 'e':
        break
    if algorithm.lower() == 'r':
        real_ttb.cleanPath()
        continue

    if algorithm != '7':
        process_dir = f"{path}/saved_models/{translator[int(algorithm)][0]}_{config['dense_size']}_A{config['num_agents']}_S{env}_{'P' if (algorithm == '3' or algorithm == '4') else 'N'}"
        # list_dir = sorted(os.listdir(process_dir))
        list_dir = "local_episode_1000_reward_200.000000.pt"
        model_fn = f"{process_dir}/{list_dir}"
        #for i, l in enumerate(list_dir):
        #    print(i, l)

        #print('Loaded:', list_dir[0])

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

    local_episode = 0
    while local_episode < episodes:
        if local_episode == 0:
            quit = input("Press [Enter] to start the test or press [q] to quit...")
        else:
            quit = input("Press [Enter] to continue to the next episode or press [q] to quit...")
        if quit.lower() == 'q':
            break

        real_ttb.cleanPath()
        episode_reward = 0
        lidar_list = list()
        num_steps = 0
        local_episode += 1
        ep_start_time = time.time()
        done = False
        while True:
            start = time.time()
            print('Num steps:', num_steps)
            if state is not None:
                for s in range(len(state)):
                    if state[s] > 2.5:
                        state[s] = 2.5

            print('State:', state)
            # state[:24] = list(reversed(state[:24]))
            # state[-2] = -state[-2]

            if algorithm != '7':
                if algorithm == '2' or algorithm == '4':
                    action, _, _, _, _, _, _, _ = actor.forward(torch.Tensor(state).to(config['device']), deterministic=True)
                else:
                    action = actor.get_action(np.array(state))
                action = action.detach().cpu().numpy().flatten()
            else:
                action = b2.get_action(state)
            action[0] = np.clip(action[0], action_low[0], action_high[0])
            action[1] = np.clip(action[1], action_low[1], action_high[1])

            print('Action:', action)
            # action[0] /= 2
            # action[1] /= 1.2
            _, _, _, _ = env_real.step(action=action)
            done = False
            reward = 0
            for i in range(len(state[:24])):
                if state[i] == 0.0:
                    state[i] = 1.0
            if state[-1] < 0.3:
                done = True
                reward = 20
            if 0.1 < min(state[0:24]) < 0.14:
                done = True
                reward = -200
            episode_reward += reward

            scan = state[0:24]
            lidar_list.append(scan)

            print('Done:', done)
            if done or num_steps == max_steps:
                break
            else:
                num_steps += 1

            print('Step timing:', time.time() - start)
            fps = round(1 / (time.time() - start))
            print('FPS:', fps)

        # Log metrics
        episode_timing = time.time() - ep_start_time
        print(f"Agent: [Test] Episode: [{local_episode}/{episodes}] Reward: [{episode_reward}/20] "
              f"Steps: [{num_steps}/{max_steps}] Episode Timing: {round(episode_timing, 2)}s")

        # Save log file
        values = [episode_reward, episode_timing, local_episode, num_steps, real_ttb.pts, lidar_list]
        with open(path_results + '/{}_{}_S{}_episode{}'.format(translator[int(algorithm)][0], translator[int(algorithm)][1], env, local_episode), "wb") as fp:
            pickle.dump(values, fp)
    print('Episode done!')
