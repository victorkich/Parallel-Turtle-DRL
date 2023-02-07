#! /usr/bin/env python3

import rospy
from utils import range_finder as rf
import gym_turtlebot3
from models import PolicyNetwork, TanhGaussianPolicy, DiagGaussianActor
from utils.defisheye import Defisheye
from algorithms.bug2 import BUG2
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
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
frame = None
font = cv2.FONT_HERSHEY_SIMPLEX
outfile = TemporaryFile()

# Hyper parameters
episodes = 12
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
real_ttb = rf.RealTtb(config, path, output=(720, 720))
defisheye = Defisheye(dtype='linear', format='fullframe', fov=155, pfov=110)


def getImage(image):
    global state
    global frame
    try:
        lidar = rospy.wait_for_message('/scan', LaserScan, timeout=1)
    except:
        pass
    image = bridge.compressed_imgmsg_to_cv2(image)
    # frame = imutils.rotate_bound(frame, 2)
    image = defisheye.convert(image)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    angle = distance = None
    try:
        lidar = np.array(lidar.ranges)
        lidar = np.array([min(lidar[[i - 1, i, i + 1]]) for i in range(7, 361, 15)]).squeeze()
        angle, distance, frame = real_ttb.get_angle_distance(image, lidar, green_magnitude=1.0)
        distance += 0.10
    except:
        pass

    if not angle is None and not distance is None:
        state = np.hstack([lidar, angle, distance])

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

print('PASSOU AQUI')
sub_image = rospy.Subscriber('/camera_2/image_raw/compressed', CompressedImage,  getImage, tcp_nodelay=True, queue_size=1, buff_size=2**26)

RECORD = True
awn_record = input("Do you wanna record your tests? [Y/n]\n")
if awn_record == 'n':
    RECORD = False
if RECORD:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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
        algorithm = input('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->DDPG-P | 8->SAC-P | 9->BUG2 | '
                          'e->exit | r->reset\n')
    if algorithm.lower() == 'e':
        break
    if algorithm.lower() == 'r':
        real_ttb.cleanPath()
        continue

    if algorithm != '9':
        process_dir = f"{path}/saved_models/{translator[int(algorithm)][0]}_{config['dense_size']}_A{config['num_agents']}_S{env}_{'P' if any(algorithm == algorithms_sel[[2, 3, 6, 7]]) else 'N'}"
        list_dir = sorted(os.listdir(process_dir))[-2]
        model_fn = f"{process_dir}/{list_dir}"
        # for i, l in enumerate(list_dir):
        #    print(i, l)

        print('Loaded:', list_dir)

        # Loading neural network model
        if any(algorithm == algorithms_sel[[0, 2, 4, 6]]):
            actor = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], device=config['device'])
        elif any(algorithm == algorithms_sel[[1, 3]]):
            actor = TanhGaussianPolicy(config=config, obs_dim=config['state_dim'], action_dim=config['action_dim'],
                                       hidden_sizes=[config['dense_size'], config['dense_size']])
        elif any(algorithm == algorithms_sel[5, 7]):
            actor = DiagGaussianActor(config['state_dim'], config['action_dim'], config['dense_size'], 2,
                                      [-config['max_action'], config['max_action']])
        try:
            actor.load_state_dict(torch.load(model_fn, map_location=config['device']))
        except:
            actor = torch.load(model_fn)
            actor.to(config['device'])
        actor.eval()
    else:
        b2 = BUG2()

    slots = [None] * episodes
    while True:
        print('Current Slots:')
        for i, slot in enumerate(slots):
            print(i, '- slot:', slot)
        value = input("Press [slot number] to start the test or press [q] to quit...\n")
        if value.lower() == 'q':
            break

        print(state)

        real_ttb.cleanPath()
        episode_reward = 0
        lidar_list = list()
        num_steps = 0
        ep_start_time = time.time()
        done = False
        if RECORD:
            out = cv2.VideoWriter(path_results + '/{}_{}_S{}_episode{}.mp4'.format(translator[int(algorithm)][0],
                                                                                   translator[int(algorithm)][1],
                                                                                   env, value), fourcc, 20.0, (720, 720))
        while True:
            start = time.time()
            if RECORD:
                out.write(frame)
            print('Num steps:', num_steps)
            # if state is not None:
            #    for s in range(len(state)):
            #        if state[s] > 2.5:
            #            state[s] = 2.5

            print('State:', state)
            # state[:24] = list(reversed(state[:24]))
            # state[-2] = -state[-2]

            if algorithm != '9':
                if any(algorithm == algorithms_sel[[0, 2, 4, 6]]):
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
            if RECORD:
                out.write(frame)
            done = False
            reward = 0
            # for i in range(len(state[:24])):
            #    if state[i] == 0.0:
            #        state[i] = 1.0
            if state[-1] < 0.3:
                done = True
                reward = 200
            if 0.1 < min(state[0:24]) < 0.14:
                done = True
                reward = -20
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
        print(f"Agent: [Test] Episode: [{value}/{episodes}] Reward: [{episode_reward}/20] "
              f"Steps: [{num_steps}/{max_steps}] Episode Timing: {round(episode_timing, 2)}s")

        # Save log file
        values = [episode_reward, episode_timing, value, num_steps, real_ttb.pts, lidar_list]
        with open(path_results + '/{}_{}_S{}_episode{}.pkl'.format(translator[int(algorithm)][0],
                                                                   translator[int(algorithm)][1], env, value), "wb") as fp:
            pickle.dump(values, fp)
        if RECORD:
            out.release()
        slots[int(value)] = 'Reward: [{episode_reward}/20] Steps: [{num_steps}/{max_steps}] ' \
                            'Episode Timing: {round(episode_timing, 2)}s'
    print('Episode done!')
