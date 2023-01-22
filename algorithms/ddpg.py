from utils.utils import empty_torch_queue
from models import Critic
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import enlighten
import torch
import queue
import time
import os


class LearnerDDPG(object):
    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        self.config = config
        self.update_iteration = config['update_agent_ep']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.device = config['device']
        self.save_dir = log_dir
        self.learner_w_queue = learner_w_queue
        self.action_high = [1.5, 0.12]
        self.priority_epsilon = config['priority_epsilon']
        self.prioritized_replay = config['replay_memory_prioritized']
        self.recurrent = config['recurrent_policy']
        self.model = config['model']
        self.env_stage = config['env_stage']
        self.num_train_steps = config['num_steps_train']  # number of episodes from all agents

        self.actor = policy_net
        self.actor_target = target_policy_net
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])

        self.critic = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic_target = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

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

        # -------------------

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_obs)[0]
        next_state_action_values = self.critic_target(next_obs, next_action_batch.detach())

        # Compute the target
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)
        expected_values = rewards + (1.0 - terminals) * self.gamma * next_state_action_values

        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(obs, actions)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())

        # Update priorities in buffer
        if self.prioritized_replay:
            td_error = value_loss.cpu().detach().numpy().flatten()
            weights_update = np.abs(td_error) + self.priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            value_loss = value_loss * torch.tensor(weights).float().to(self.device)
            value_loss = value_loss.mean()

        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(obs, self.actor(obs)[0])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ---------------------

        """
        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state)[0])
        target_Q = reward + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        critic_loss = critic_loss.mean()

        # Optimize the critic
        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)[0])  # .mean()

        # Optimize the actor
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        """

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                self.learner_w_queue.put(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[3] = policy_loss
            logs[4] = value_loss
            logs[5] = time.time() - update_time

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
            except queue.Empty:
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

    """def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
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
        print("Exit learner.")"""
