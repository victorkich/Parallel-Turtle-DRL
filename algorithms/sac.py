from utils.utils import empty_torch_queue
from models import DoubleQCritic
from torch.distributions import Normal
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import enlighten
import torch
import time


class LearnerSAC(object):
    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        self.config = config
        self.update_iteration = config['update_agent_ep']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.device = config['device']
        self.save_dir = log_dir
        self.learner_w_queue = learner_w_queue
        self.prioritized_replay = config['replay_memory_prioritized']
        self.priority_epsilon = config['priority_epsilon']
        self.model = config['model']
        self.env_stage = config['env_stage']
        self.num_train_steps = config['num_steps_train']  # number of episodes from all agents
        self.recurrent = config['recurrent_policy']

        self.actor = policy_net
        self.actor_target = target_policy_net
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])

        self.critic = DoubleQCritic(config['state_dim'], config['action_dim'], config['dense_size'], 1)
        # self.critic = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])

        # self.Q_net = Q(config['state_dim'], config['action_dim'], hidden=config['dense_size']).to(self.device)
        # self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=config['actor_learning_rate'])
        self.critic_target = DoubleQCritic(config['state_dim'], config['action_dim'], config['dense_size'], 1)
        # self.critic_target = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)

        self.learnable_temperature = config['use_automatic_entropy_tuning']
        self.target_entropy = -2
        self.init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config['actor_learning_rate'])

        self.critic_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()
        self.num_training = 0

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action.item()

    def alpha(self):
        return self.log_alpha.exp()

    def get_action_log_prob(self, state):
        min_Val = torch.tensor(1e-7).float()
        batch_mu, batch_log_sigma, _ = self.actor(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def _update_step(self, batch, replay_priority_queue, update_step, logs):
        update_time = time.time()

        # Sample replay buffer
        obs, actions, rewards, next_obs, terminals, gamma, weights, inds = batch

        obs = np.asarray(obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_obs = np.asarray(next_obs)
        terminals = np.asarray(terminals)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()

        obs = torch.from_numpy(obs).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        terminals = torch.from_numpy(terminals).float().to(self.device)

        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)

        """
        # Compute the target Q value
        target_value = self.critic_target(next_obs, self.actor(obs)[0])
        next_q_value = rewards + (1.0 - terminals) * self.gamma * target_value
        excepted_value, _, _ = self.actor(obs)
        excepted_Q = self.Q_net(obs, actions)

        # Get current Q estimate
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(obs)
        excepted_new_Q = self.Q_net(obs, sample_action)
        next_value = excepted_new_Q - log_prob

        # Compute critic loss
        critic_loss = self.critic_criterion(excepted_value, next_value.detach())  # J_V
        critic_loss = critic_loss.mean()

        # Compute Q loss. Single Q_net this is different from original paper
        Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach())  # J_Q

        if self.prioritized_replay:
            td_error = Q_loss.cpu().detach().numpy().flatten()
            weights_update = np.abs(td_error) + self.priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            Q_loss = Q_loss * torch.tensor(weights).float().to(self.device)
            Q_loss = Q_loss.mean()

        # Compute actor loss
        log_policy_target = excepted_new_Q - excepted_value
        pi_loss = log_prob * (log_prob - log_policy_target).detach()
        pi_loss = pi_loss.mean()

        # Optimize the critic. Mini batch gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()
        """

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = rewards + ((1.0 - terminals) * self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.prioritized_replay:
            td_error = critic_loss.cpu().detach().numpy().flatten()
            weights_update = np.abs(td_error) + self.priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            critic_loss = critic_loss * torch.tensor(weights).float().to(self.device)
            critic_loss = critic_loss.mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        # soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[3] = actor_loss
            logs[4] = critic_loss
            logs[5] = time.time() - update_time

        self.num_training += 1

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
        torch.set_num_threads(4)
        time.sleep(2)

        manager = enlighten.get_manager()
        status_format = '{program}{fill}Stage: {stage}{fill} Status {status}'
        algorithm = f"{self.model}-{'P' if self.prioritized_replay else 'N'}-{'LSTM' if self.recurrent else ''}"
        status_bar = manager.status_bar(status_format=status_format, color='bold_slategray', program=algorithm, stage=str(self.env_stage), status='Training')
        ticks = manager.counter(total=self.num_train_steps, desc="Training step", unit="ticks", color="red")
        while update_step.value <= self.num_train_steps:
            try:
                batch = batch_queue.get_nowait()
            except:
                ticks.update(0)
                time.sleep(0.01)
                continue

            self._update_step(batch, replay_priority_queue, update_step, logs)
            ticks.update(1)
            with update_step.get_lock():
                update_step.value += 1

        with training_on.get_lock():
            training_on.value = 0

        status_bar.update(status='Ending')
        empty_torch_queue(self.learner_w_queue)
        empty_torch_queue(replay_priority_queue)
        torch.cuda.empty_cache()
        time.sleep(5)
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")
        os.system("kill $(ps aux | grep gzclient | grep -v grep | awk '{print $2}')")
        os.system("kill $(ps aux | grep gzserver | grep -v grep | awk '{print $2}')")
        print("Exit learner.")
