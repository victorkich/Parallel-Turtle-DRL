#! /usr/bin/env python3

import rospy
from utils.utils import OUNoise, empty_torch_queue, test_goals
from utils import range_finder as rf
from collections import deque
import gym_turtlebot3
import numpy as np
import torch
import time
import gym
import os


class Agent(object):
    def __init__(self, config, policy, global_episode, global_step, n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
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
        print("Started agent", n_agent, "using", self.actor.device)

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

    def run(self, training_on, replay_queue, learner_w_queue, logs):
        time.sleep(1)
        os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + self.n_agent)
        rospy.init_node(self.config['env_name'].replace('-', '_') + "_w{}".format(self.n_agent))
        goal = None
        if self.config['test']:
            goal = [test_goals(self.local_episode)]
        env = gym.make(self.config['env_name'], env_stage=self.config['env_stage'], observation_mode=0, continuous=True, goal_list=goal)
        time.sleep(1)

        if self.config['test_real']:
            real_ttb = rf.RealTtb(self.config, self.log_dir, output=(720, 480))

        best_reward = -float("inf")
        rewards = []
        while self.local_episode <= self.config['num_episodes'] if not self.config['test'] else self.config['test_trials']:
            if self.config['test_real']:
                input("Press Enter to continue to the next episode...")
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
            while not done:
                if self.config['test_real']:
                    state = real_ttb.get_angle_distance(state, 1.5)
                action = self.actor.get_action(torch.Tensor(state).to(self.config['device']) if (not self.config[
                    'test'] and not self.config['model'] == 'D4PG') or self.config['model'] == 'DSAC' else np.array(
                    state))
                if self.agent_type == "exploration" and not self.config['model'] == 'DSAC':
                    action = action.squeeze(0)
                    action = self.ou_noise.get_action(action, num_steps)
                else:
                    action = action.detach().cpu().numpy().flatten()
                    action[0] = np.clip(action[0], self.action_low[0], self.action_high[0])
                    action[1] = np.clip(action[1], self.action_low[1], self.action_high[1])

                next_state, reward, done, info = env.step(action, exploitation=True if self.agent_type == "exploitation" else False)  # test_real=self.config['test_real']
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
                                    replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done,
                                                             gamma])
                                except:
                                    pass
                    break

                num_steps += 1
                with self.global_step.get_lock():
                    self.global_step.value += 1

                if self.config['test']:
                    if self.config['test_real']:
                        position = env.get_position()  # Get x and y turtlebot position to compute test charts
                        # scan = env.get_scan()
                        logs[3] = position[0]
                        logs[4] = position[1]
                    else:
                        position = env.get_position()  # Get x and y turtlebot position to compute test charts
                        # scan = env.get_scan()
                        logs[3] = position[0]
                        logs[4] = position[1]

            with self.global_episode.get_lock():
                self.global_episode.value += 1

            # Log metrics
            episode_timing = time.time() - ep_start_time
            print(
                f"Agent: [{self.n_agent}/{self.config['num_agents'] - 1}] Episode: [{self.local_episode}/{self.config['test_trials']}] "
                f"Reward: [{episode_reward}/200] Step: {self.global_step.value} Episode Timing: {round(episode_timing, 2)}s")
            aux = 6 + self.n_agent * 3
            with logs.get_lock():
                if not self.config['test']:
                    logs[aux] = episode_reward
                    logs[aux + 1] = episode_timing
                    logs[aux + 2] = self.local_episode
                else:
                    logs[0] = episode_reward
                    logs[1] = episode_timing
                    logs[2] = self.local_episode

            # Saving agent
            if not self.config['test']:
                reward_outperformed = episode_reward - best_reward > self.config["save_reward_threshold"]
                time_to_save = self.local_episode % self.num_episode_save == 0
                if self.agent_type == "exploitation" and (time_to_save or reward_outperformed):
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

                rewards.append(episode_reward)
                if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                    self.update_actor_learner(learner_w_queue, training_on)

        if not self.config['test']:
            empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/{self.config['model']}_{self.config['dense_size']}_A{self.config['num_agents']}_S{self.config['env_stage']}_{'P' if self.config['replay_memory_prioritized'] else 'N'}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)
