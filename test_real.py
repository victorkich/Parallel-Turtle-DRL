#! /usr/bin/env python3

import rospy
from utils import range_finder as rf
import gym_turtlebot3
from models import PolicyNetwork, TanhGaussianPolicy
from algorithms.vector_field import VectorField
from utils import unfish
from math import isnan
import pandas as pd
import numpy as np
import torch
import yaml
import time
import gym
import os

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
# os.environ['ROS_MASTER_URI'] = "http://192.168.31.225:11311"
rospy.init_node(config['env_name'].replace('-', '_') + "_test_real")
env_real = gym.make(config['env_name'], env_stage=env.lower(), observation_mode=0, continuous=True)
real_ttb = rf.RealTtb(config, path, output=(1280, 720))
state = env_real.reset(test_real=True)
camera_matrix, coeffs = unfish.calibrate(state[1])
rf.setCamSettings(camera_matrix=camera_matrix, coeffs=coeffs)

path_results = path + '/real_results'
if not os.path.exists(path_results):
    os.makedirs(path_results)
if not os.path.exists(path_results+'/real_results_S{}.csv'.format(env)):
    print('File real_results_S{}.csv not found!\nGeneraring a new file real_results_S{}.csv ...'.format(env, env))
    df = pd.DataFrame({'PDDRL': [], 'PDSRL': [], 'PDDRL-P': [], 'PDSRL-P': [], 'DDPG': [], 'SAC': [], 'Vector Field': []})
    df.to_csv(path_results+'/real_results_S{}.csv'.format(env))
else:
    print('File real_results_S{}.csv found!\nLoading data from real_results_S{}.csv...'.format(env, env))
    df = pd.read_csv(path_results+'/real_results_S{}.csv'.format(env))
data = df.to_dict()

time.sleep(1)
translator = {1: ['PDDRL', 'N'], 2: ['PDSRL', 'N'], 3: ['PDDRL', 'P'], 4: ['PDSRL', 'P'], 5: ['DDPG', 'N'],
              6: ['SAC', 'N'], 7: ['Vector_Field', 'N']}
algorithms_sel = np.array(['1', '2', '3', '4', '5', '6', '7', 'e', 'r'])
algorithm = ""
while True:
    while not any(algorithm.lower() == algorithms_sel):
        print('Choose the algorithm or exit the test:')
        algorithm = input('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->Vector Field | '
                          'e->exit | r->reset\n')
    if algorithm.lower() == 's':
        break
    if algorithm.lower() == 'r':
        reset = input('Do you want to reset any test results? [y/n]\n')
        if reset.lower() == 'y':
            reset = input('Do you want to reset all the test results? [y/n]\n')
            if reset.lower() == 'y':
                df = pd.DataFrame({'PDDRL': [], 'PDSRL': [], 'PDDRL-P': [], 'PDSRL-P': [], 'DDPG': [], 'SAC': [], 'Vector Field': []})
                df.to_csv(path_results + '/real_results.csv')
            else:
                column = None
                while not any(column == np.arange(7)+1):
                    print('1->PDDRL | 2->PDSRL | 3->PDDRL-P | 4->PDSRL-P | 5->DDPG | 6->SAC | 7->Vector Field')
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
            actor.load_state_dict(torch.load(model_fn))
        except:
            actor = torch.load(model_fn)
        actor.eval()
    else:
        vf = VectorField()

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
        lidar = list()
        num_steps = 0
        local_episode += 1
        ep_start_time = time.time()
        state = env_real.reset(test_real=True)
        done = False
        while True:
            print('Num steps:', num_steps)
            state = real_ttb.get_angle_distance(state, 1.0)
            if algorithm != '7':
                action = actor.get_action(torch.Tensor(state).to(config['device']) if config['model'] == 'DSAC' else np.array(state))
                if not config['model'] == 'DSAC':
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()
                action[0] = np.clip(action[0], action_low[0], action_high[0])
                action[1] = np.clip(action[1], action_low[1], action_high[1])
            else:
                action = vf.get_action(state)

            next_state = env_real.step(action, test_real=True)
            reward, done = env_real.get_done_reward()
            episode_reward += reward
            state = next_state

            position = env_real.get_position()  # Get x and y turtlebot position to compute test charts
            scan = env_real.get_scan()
            xy.append(position)
            lidar.append(scan)

            if done or num_steps == max_steps:
                break
            else:
                num_steps += 1

        # Log metrics
        episode_timing = time.time() - ep_start_time
        print(f"Agent: [Test] Episode: [{local_episode}/{episodes}] Reward: [{episode_reward}/200] "
              f"Steps: [{num_steps}/{max_steps}] Episode Timing: {round(episode_timing, 2)}s")

        # Save csv file
        values = [episode_reward, episode_timing, local_episode, num_steps, xy, lidar]
        data[list(data.keys())[int(algorithm) - 1]] = filter(lambda k: not isnan(k), data[list(data.keys())[int(algorithm) - 1]])
        data[list(data.keys())[int(algorithm) - 1]].append(values)
        df = pd.DataFrame.from_dict(data, orient='index').T
        df.to_csv(path_results + '/real_results_S{}.csv'.format(env))
    print('Done!')
