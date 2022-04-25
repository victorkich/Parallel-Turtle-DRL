#! /usr/bin/env python3

import rospy
from utils.utils import OUNoise, empty_torch_queue, test_goals
from colorama import init as colorama_init
from collections import deque
from colorama import Fore
import gym_turtlebot3
import numpy as np
import torch
import time
import sys
import gym
import os
gym.logger.set_level(40)


class Agent(object):
    def __init__(self, config, policy, global_episode, global_step, n_agent=0, agent_type='exploration', log_dir=''):
        colorama_init(autoreset=True)
        self.colors = dict(Fore.__dict__.items())
        self.color = list(self.colors.keys())[n_agent + 1]

        print(self.colors[self.color] + f"Initializing agent {n_agent}...")
        self.config = config
        self.action_low = [-1.5, -0.1]
        self.action_high = [1.5, 0.12]
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']  # maximum number of steps per episode
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.global_step = global_step
        self.local_episode = 0
        self.log_dir = log_dir
        # number of future steps to collect experiences for N-step returns
        self.n_step_returns = config['n_step_return']
        self.discount_rate = config['discount_rate']  # Discount rate (gamma) for future rewards
        # agent gets latest parameters from learner every update_agent_ep episodes
        self.update_agent_ep = config['update_agent_ep']

        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        # Create environment
        self.ou_noise = OUNoise(dim=config['action_dim'], low=self.action_low, high=self.action_high)
        self.ou_noise.reset()

        self.actor = policy
        print(self.colors[self.color] + f"Started agent {n_agent} using {config['device']}")

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

    def run(self, training_on, replay_queue, learner_w_queue, update_step, logs):

        time.sleep(1)
        os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + self.n_agent)
        rospy.init_node(self.config['env_name'].replace('-', '_') + "_w{}".format(self.n_agent))
        goal = None
        if self.config['test']:
            goal = [test_goals(self.local_episode)]
        env = gym.make(self.config['env_name'], env_stage=self.config['env_stage'], observation_mode=0, continuous=True, goal_list=goal)
        time.sleep(1)

        best_reward = -float("inf")
        rewards = []
        while training_on.value if not self.config['test'] else (self.local_episode <= self.config['test_trials']):
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            ep_start_time = time.time()
            goal = None
            if self.config['test']:
                goal = [test_goals(self.local_episode)]
                print("New Goal:", goal)
            state = env.reset(new_random_goals=True if not self.config['test'] else False, goal=goal)
            if not self.config['test']:
                self.exp_buffer.clear()
                self.ou_noise.reset()
            done = False
            if self.config['recurrent_policy']:
                sequence_replay_buffer = []

            h_0 = None
            c_0 = None
            while not done:
                for s in range(len(state)):
                    if state[s] > 2.5:
                        state[s] = 2.5

                if self.config['model'] == 'PDSRL' or self.config['model'] == 'SAC':
                    action, _, _, _, _, _, _, _ = self.actor.forward(torch.Tensor(state).to(self.config['device']),
                                                                     deterministic=True if self.agent_type == "exploitation" else False)
                    action = action.detach().cpu().numpy().flatten()
                else:
                    action, (h_0, c_0) = self.actor.get_action(np.array(state), h_0=h_0, c_0=c_0)
                    if self.agent_type == "exploration":
                        action = action.squeeze(0)
                        action = self.ou_noise.get_action(action, num_steps).flatten()
                    else:
                        action = action.detach().cpu().numpy().flatten()

                action[0] = np.clip(action[0], self.action_low[0], self.action_high[0])
                action[1] = np.clip(action[1], self.action_low[1], self.action_high[1])

                next_state, reward, done, info = env.step(action)
                if reward > 200:
                    reward = 200
                episode_reward += reward

                if not self.config['test']:
                    self.exp_buffer.append((state, action, reward))

                    # We need at least N steps in the experience buffer before we can compute Bellman
                    # rewards and add an N-step experience to replay memory
                    if len(self.exp_buffer) >= self.config['n_step_return']:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        # We want to fill buffer only with form explorator
                        if self.agent_type == "exploration":
                            try:
                                if self.config['recurrent_policy'] and len(sequence_replay_buffer) < self.config['sequence_size']:
                                    sequence_replay_buffer.append([state_0, action_0, discounted_reward, next_state, done, gamma, h_0, c_0])
                                elif self.config['recurrent_policy']:
                                    replay_queue.put_nowait([[srb[i] for srb in sequence_replay_buffer] for i in range(8)])
                                    sequence_replay_buffer = []
                                else:
                                    replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                            except:
                                pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    if not self.config['test']:
                        while len(self.exp_buffer) != 0:
                            state_0, action_0, reward_0 = self.exp_buffer.popleft()
                            discounted_reward = reward_0
                            gamma = self.config['discount_rate']
                            for (_, _, r_i) in self.exp_buffer:
                                discounted_reward += r_i * gamma
                                gamma *= self.config['discount_rate']
                            if self.agent_type == "exploration":
                                try:
                                    if self.config['recurrent_policy']:
                                        while len(sequence_replay_buffer) < self.config['sequence_size']:
                                            sequence_replay_buffer.append([state_0, action_0, discounted_reward,
                                                                           next_state, done, gamma, h_0, c_0])
                                        replay_queue.put_nowait([[srb[i] for srb in sequence_replay_buffer] for i in range(8)])
                                    else:
                                        replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma, h_0, c_0])
                                except:
                                    pass
                    break

                num_steps += 1
                with self.global_step.get_lock():
                    self.global_step.value += 1

                if self.config['test']:
                    position = env.get_position()  # Get x and y turtlebot position to compute test charts
                    logs[3] = position[0]
                    logs[4] = position[1]

            with self.global_episode.get_lock():
                self.global_episode.value += 1

            # Log metrics
            episode_timing = time.time() - ep_start_time
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line
            print(self.colors[self.color] + f"Approach: [{self.config['model']}-{'P' if self.config['replay_memory_prioritized'] else 'N'}] "
                  f"Agent: [{self.n_agent + 1}/{self.config['num_agents']}] Episode: [{self.local_episode}/"
                  f"{self.config['test_trials'] if self.config['test'] else self.config['num_episodes']}] Reward: "
                  f"[{episode_reward}/200] Episode Timing: {round(episode_timing, 2)}s", '\n')
            aux = 6 + self.n_agent * 3
            with logs.get_lock():
                if not self.config['test']:
                    logs[aux] = episode_reward
                    logs[aux+1] = episode_timing
                    logs[aux+2] = self.local_episode
                else:
                    logs[0] = episode_reward
                    logs[1] = episode_timing
                    logs[2] = self.local_episode

            # Saving agent
            if not self.config['test']:
                time_to_save = self.local_episode % self.num_episode_save == 0
                if self.agent_type == "exploitation" and (time_to_save or self.global_step.value >= self.config['num_steps_train']):
                    self.save(f"step_{self.global_step.value}_episode_{self.local_episode}")

                rewards.append(episode_reward)
                if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                    self.update_actor_learner(learner_w_queue, training_on)

        if not self.config['test']:
            empty_torch_queue(replay_queue)
        rospy.signal_shutdown(f"Agent {self.n_agent} done.")
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/{self.config['model']}_{self.config['dense_size']}_A{self.config['num_agents']}_S" \
                      f"{self.config['env_stage']}_{'P' if self.config['replay_memory_prioritized'] else 'N'}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pth"
        torch.save(self.actor.state_dict(), model_fn)
