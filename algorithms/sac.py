from utils.utils import empty_torch_queue
from models import Critic, Q
from torch.distributions import Normal
import torch.optim as optim
import torch.nn as nn
import numpy as np
import queue
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

        self.actor = policy_net
        self.actor_target = target_policy_net
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])
        self.critic = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])

        self.Q_net = Q(config['state_dim'], config['action_dim'], hidden=config['dense_size']).to(self.device)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=config['actor_learning_rate'])
        self.critic_target = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)

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
        # x, u, r, y, d, gamma, weights, inds = replay_buffer
        # x, u, r, y, d, _ = replay_buffer

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

        # state = torch.FloatTensor(x).to(self.device)
        # next_state = torch.FloatTensor(y).to(self.device)
        # action = torch.FloatTensor(u).to(self.device)
        # reward = torch.FloatTensor(r).to(self.device)
        # done = torch.FloatTensor(1 - d).to(self.device)

        # -----------------------

        # -------------------------------------
        """
        # Compute the target Q value
        target_value = self.critic_target(next_obs, self.actor(obs)[0]).squeeze(-1)
        next_q_value = rewards + (1 - terminals) * self.config['discount_rate'] * target_value
        next_q_value = next_q_value.unsqueeze(-1)
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
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
        self.Q_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        pi_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        """

        # soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param * (1 - self.config['tau']) + param * self.config['tau'])

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param * (1 - self.config['tau']) + param * self.config['tau'])

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                self.learner_w_queue.put(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[3] = pi_loss
            logs[4] = Q_loss
            logs[5] = time.time() - update_time

        self.num_training += 1

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
        torch.set_num_threads(4)
        while global_episode.value <= self.config['num_agents'] * self.config['num_episodes']:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            self._update_step(batch, replay_priority_queue, update_step, logs)
            with update_step.get_lock():
                update_step.value += 1

            if update_step.value % 10000 == 0:
                print("Training step ", update_step.value)

        with training_on.get_lock():
            training_on.value = 0

        empty_torch_queue(self.learner_w_queue)
        empty_torch_queue(replay_priority_queue)
        print("Exit learner.")
