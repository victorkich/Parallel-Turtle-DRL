#! /usr/bin/env python3

import rospy
from utils.utils import OUNoise, empty_torch_queue
from collections import deque
import gym_turtlebot3
import pandas as pd
import numpy as np
import torch
import time
import gym
import os
env_name = 'TurtleBot3_Circuit_Simple-v0'


class Agent(object):
    def __init__(self, policy, global_episode, n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
        state_dim = 26
        action_dim = 2
        self.action_low = [-1.5, -0.1]
        self.action_high = [1.5, 0.12]
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = 500  # maximum number of steps per episode
        self.num_episode_save = 100
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir
        self.n_step_returns = 5  # number of future steps to collect experiences for N-step returns
        self.discount_rate = 0.99  # Discount rate (gamma) for future rewards
        self.update_agent_ep = 1  # agent gets latest parameters from learner every update_agent_ep episodes

        # Create environment
        self.ou_noise = OUNoise(dim=action_dim, low=self.action_low, high=self.action_high)
        self.ou_noise.reset()

        self.actor = policy
        print("Agent ", n_agent, self.actor.device)

    def update_actor_learner(self, learner_w_queue, training_on):
        """Update local actor to the actor from learner. """
        if not training_on.value:
            return
        try:
            source = learner_w_queue.get_nowait()
        except:
            return
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)
        del source

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        time.sleep(1)
        os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + self.n_agent)
        rospy.init_node(env_name.replace('-', '_') + "_w{}".format(self.n_agent))
        env = gym.make(env_name, observation_mode=0, continuous=True,
                       goal_list=[(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (1.0, 1.0)])
        time.sleep(1)
        df = pd.DataFrame(columns=['episode', 'reward'])
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            ep_start_time = time.time()
            state = env.reset(new_random_goals=True)
            self.ou_noise.reset()
            done = False
            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()
                    action[0] = np.clip(action[0], self.action_low[0], self.action_high[0])
                    action[1] = np.clip(action[1], self.action_low[1], self.action_high[1])
                next_state, reward, done, info = env.step(action)

                episode_reward += reward

                # state = env.normalise_state(state)
                # reward = env.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.n_step_returns:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.discount_rate
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.discount_rate
                    # We want to fill buffer only with form explorator
                    if self.agent_type == "exploration":
                        try:
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                        except:
                            pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.discount_rate
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.discount_rate
                        if self.agent_type == "exploration":
                            try:
                                replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                            except:
                               pass
                    break

                num_steps += 1

            df = df.append({'episode': self.local_episode, 'reward': episode_reward}, ignore_index=True)
            df.to_csv('log/agent_{}.csv'.format(self.n_agent))

            print('Episode:', self.local_episode, 'Agent:', self.n_agent, 'Reward:', episode_reward)

            # Saving agent
            time_to_save = self.local_episode % self.num_episode_save == 0
            if self.n_agent == 0 and time_to_save:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % self.update_agent_ep == 0:
                self.update_actor_learner(learner_w_queue, training_on)

        empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)
