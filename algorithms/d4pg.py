from utils.l2_projection import _l2_project
from utils.utils import empty_torch_queue
from models2 import ValueNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np
import enlighten
import queue
import torch
import time
import os


class LearnerD4PG(object):
    """Policy and value network update routine. """
    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        self.config = config
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        self.n_step_return = config['n_step_return']
        self.v_min = config['v_min']  # lower bound of critic value output distribution
        self.v_max = config['v_max']  # upper bound of critic value output distribution
        self.num_atoms = config['num_atoms']  # number of atoms in output layer of distributed critic
        self.device = config['device']
        self.max_steps = config['max_ep_length']  # maximum number of steps per episode
        self.num_train_steps = config['num_steps_train']  # number of episodes from all agents
        self.batch_size = config['batch_size']
        self.tau = config['tau']  # parameter for soft target network updates
        self.gamma = config['discount_rate']  # Discount rate (gamma) for future rewards
        self.log_dir = log_dir
        self.prioritized_replay = config['replay_memory_prioritized']
        self.learner_w_queue = learner_w_queue
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.recurrent = config['recurrent_policy']
        self.sequence_size = config['sequence_size']
        self.priority_epsilon = config['priority_epsilon']
        self.model = config['model']
        self.env_stage = config['env_stage']

        # Value and policy nets
        self.value_net = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], self.v_min,
                                      self.v_max, self.num_atoms, device=self.device,
                                      recurrent=self.recurrent)
        self.policy_net = policy_net
        self.target_value_net = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'],
                                             self.v_min, self.v_max, self.num_atoms, device=self.device,
                                             recurrent=self.recurrent)
        self.target_policy_net = target_policy_net

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.BCELoss(reduction='none')

    def _update_step(self, batch, replay_priority_queue, update_step, logs):
        update_time = time.time()

        state, action, reward, next_state, done, gamma, h_0, c_0, weights, inds = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()
        h_0 = np.asarray(h_0)
        c_0 = np.asarray(c_0)

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        h_0 = torch.from_numpy(h_0).float().to(self.device)
        c_0 = torch.from_numpy(c_0).float().to(self.device)

        # ------- Update critic -------
        # Predict next actions with target policy network
        next_action, _ = self.target_policy_net(next_state, h_0=h_0, c_0=c_0)

        # Predict Z distribution with target value network
        target_value = self.target_value_net.get_probs(next_state, next_action.detach())

        # Get projected distribution
        target_z_projected = _l2_project(next_distr_v=target_value.view(-1, 51),
                                         rewards_v=reward.view(-1, 1).squeeze(),
                                         dones_mask_t=done.view(-1, 1).squeeze(),
                                         gamma=self.gamma ** self.n_step_return,
                                         n_atoms=self.num_atoms, v_min=self.v_min, v_max=self.v_max, delta_z=self.delta_z)

        if self.recurrent:
            target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device).view(self.batch_size, -1, 51)
        else:
            target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device).view(self.batch_size, 51)

        critic_value = self.value_net.get_probs(state, action)
        critic_value = critic_value.to(self.device)
        value_loss = self.value_criterion(critic_value, target_z_projected)
        if self.recurrent:
            value_loss = torch.sum(value_loss, dim=2)
        value_loss = value_loss.mean(axis=1)

        # Update priorities in buffer
        if self.prioritized_replay:
            td_error = value_loss.cpu().detach().numpy().flatten()
            weights_update = np.abs(td_error) + self.priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            if self.recurrent:
                w_shape = weights.shape[0]
                weights = weights.reshape((w_shape, 1))
            value_loss = value_loss * torch.tensor(weights).float().to(self.device)

        # Update step
        value_loss = value_loss.mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor -----------
        policy_action, _ = self.policy_net(state, h_0=h_0, c_0=c_0)
        policy_loss = self.value_net.get_probs(state, policy_action)
        policy_loss = policy_loss * torch.from_numpy(self.value_net.z_atoms).float().to(self.device)
        policy_loss = torch.sum(policy_loss, dim=2 if self.recurrent else 1)
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[3] = policy_loss.item()
            logs[4] = value_loss.item()
            logs[5] = time.time() - update_time

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, logs):
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
