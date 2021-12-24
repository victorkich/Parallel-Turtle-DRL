#! /usr/bin/env python3

import rospy
from utils import range_finder as rf
import gym_turtlebot3
import numpy as np
import torch
import yaml
import time
import gym
import os

# Loading configs from config.yaml
path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/config.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


rospy.init_node(config['env_name'].replace('-', '_') + "_w{}".format(self.n_agent))
env = gym.make(config['env_name'], env_stage=config['env_stage'], observation_mode=0, continuous=True, goal_list=goal)
real_ttb = rf.RealTtb(config, self.log_dir, output=(720, 480))
time.sleep(1)

best_reward = -float("inf")
rewards = []
local_episode = 0
while local_episode <= config['test_trials']:
    input("Press Enter to continue to the next episode...")
    episode_reward = 0
    num_steps = 0
    local_episode += 1
    ep_start_time = time.time()
    goal = None
    state = env.reset(test_real=True)
    done = False
    while not done:
        state = real_ttb.get_angle_distance(state, 1.5)
        action = self.actor.get_action(torch.Tensor(state).to(config['device']) if config['model'] == 'DSAC' else
                 np.array(state))
        if not config['model'] == 'DSAC':
            action = action.squeeze(0)
        else:
            action = action.detach().cpu().numpy().flatten()
            action[0] = np.clip(action[0], self.action_low[0], self.action_high[0])
            action[1] = np.clip(action[1], self.action_low[1], self.action_high[1])

        next_state, reward, done, info = env.step(action)  # test_real=self.config['test_real']
        episode_reward += reward
        state = next_state

        if done or num_steps == self.max_steps:
            break

        num_steps += 1
        position = env.get_position()  # Get x and y turtlebot position to compute test charts
        # scan = env.get_scan()
        logs[3] = position[0]
        logs[4] = position[1]

    # Log metrics
    episode_timing = time.time() - ep_start_time
    print(f"Agent: [{self.n_agent}/{config['num_agents'] - 1}] Episode: [{self.local_episode}/{config['test_trials']}] "
          f"Reward: [{episode_reward}/200] Step: {self.global_step.value} Episode Timing: {round(episode_timing, 2)}s")

    logs[0] = episode_reward
    logs[1] = episode_timing
    logs[2] = self.local_episode
    rewards.append(episode_reward)

print(f"Agent {self.n_agent} done.")
