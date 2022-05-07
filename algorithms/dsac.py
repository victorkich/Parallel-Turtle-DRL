from utils.utils import empty_torch_queue, fast_clip_grad_norm, quantile_regression_loss
from models import QuantileMlp
import torch.optim as optim
import numpy as np
import enlighten
import queue
import torch
import time
import os


class LearnerDSAC(object):
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
        self.beta = config['tau']
        self.gamma = config['discount_rate']  # Discount rate (gamma) for future rewards
        self.log_dir = log_dir
        self.prioritized_replay = config['replay_memory_prioritized']
        self.learner_w_queue = learner_w_queue
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self._action_prior = config['action_prior']
        self.target_entropy = -config['action_dim']
        self.action_size = config['action_dim']
        self.state_size = config['state_dim']
        self.soft_target_tau = config['tau']  # parameter for soft target network updates
        self.target_update_period = config['update_agent_ep']
        self.num_quantiles = config['num_quantiles']
        M = config['dense_size']

        # value nets
        self.zf1 = QuantileMlp(config=config, input_size=self.state_size + self.action_size, output_size=1, num_quantiles=self.num_quantiles, hidden_sizes=[M, M], recurrent=config['recurrent_policy'])
        self.zf2 = QuantileMlp(config=config, input_size=self.state_size + self.action_size, output_size=1, num_quantiles=self.num_quantiles, hidden_sizes=[M, M], recurrent=config['recurrent_policy'])
        self.target_zf1 = QuantileMlp(config=config, input_size=self.state_size + self.action_size, output_size=1, num_quantiles=self.num_quantiles, hidden_sizes=[M, M], recurrent=config['recurrent_policy'])
        self.target_zf2 = QuantileMlp(config=config, input_size=self.state_size + self.action_size, output_size=1, num_quantiles=self.num_quantiles, hidden_sizes=[M, M], recurrent=config['recurrent_policy'])

        # policy nets
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net

        for target_param, param in zip(self.target_zf1.parameters(), self.zf1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_zf2.parameters(), self.zf2.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.use_automatic_entropy_tuning = config['use_automatic_entropy_tuning']
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -torch.tensor(np.prod(config['action_dim']).item()).to(self.device)
            self.log_alpha = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True).to(self.device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
        else:
            self.alpha = config['alpha']

        # optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.zf1_optimizer = optim.Adam(self.zf1.parameters(), lr=value_lr)
        self.zf2_optimizer = optim.Adam(self.zf2.parameters(), lr=value_lr)
        self.zf_criterion = quantile_regression_loss

        self.discount = config['discount_rate']
        self.reward_scale = config['reward_scale']
        self.clip_norm = config['clip_norm']

    def get_tau(self, actions):
        if self.config['recurrent_policy']:
            batch_size, seq_size, obs_size = actions.size()
            presum_tau = torch.zeros(batch_size, seq_size, self.num_quantiles).to(self.device) + 1. / self.num_quantiles
        else:
            batch_size, obs_size = actions.size()
            presum_tau = torch.zeros(batch_size, self.num_quantiles).to(self.device) + 1. / self.num_quantiles

        tau = torch.cumsum(presum_tau, dim=2 if self.config['recurrent_policy'] else 1)  #  (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau).to(self.device)
            if self.config['recurrent_policy']:
                tau_hat[:, :, 0:1] = tau[:, :, 0:1] / 2.
                tau_hat[:, :, 1:] = (tau[:, :, 1:] + tau[:, :, :-1]) / 2.
            else:
                tau_hat[:, 0:1] = tau[:, 0:1] / 2.
                tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def _update_step(self, batch, replay_priority_queue, update_step, logs):
        update_time = time.time()

        obs, actions, rewards, next_obs, terminals, gamma, h_0, c_0, weights, inds = batch

        obs = np.asarray(obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_obs = np.asarray(next_obs)
        terminals = np.asarray(terminals)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()
        h_0 = np.asarray(h_0)
        c_0 = np.asarray(c_0)

        obs = torch.from_numpy(obs).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        terminals = torch.from_numpy(terminals).float().to(self.device)
        h_0 = torch.from_numpy(h_0).float().to(self.device)
        c_0 = torch.from_numpy(c_0).float().to(self.device)

        # ------- Update critic -------
        # Get predicted next-state actions and Q values from target models
        new_actions, policy_mean, policy_log_std, log_pi, _, _, _, _, _ = self.policy_net(obs, h_0=h_0, c_0=c_0, reparameterize=True, return_log_prob=True)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # ------- Update ZF -------
        with torch.no_grad():
            new_next_actions, _, _, new_log_pi, _, _, _, _, _ = self.target_policy_net(next_obs, h_0=h_0, c_0=c_0, reparameterize=True, return_log_prob=True)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(new_next_actions)
            target_z1_values = self.target_zf1(next_obs, new_next_actions, next_tau_hat)
            target_z2_values = self.target_zf2(next_obs, new_next_actions, next_tau_hat)
            target_z_values = torch.min(target_z1_values, target_z2_values) - alpha * new_log_pi
            if self.config['recurrent_policy']:
                terminals = terminals.view(self.config['batch_size'], self.config['sequence_size'], 1)
                rewards = rewards.view(self.config['batch_size'], self.config['sequence_size'], 1)
                z_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z_values
            else:
                z_target = self.reward_scale * rewards.unsqueeze(1) + (1. - terminals.unsqueeze(1)) * self.discount * target_z_values

        tau, tau_hat, presum_tau = self.get_tau(actions)
        z1_pred = self.zf1(obs, actions, tau_hat)
        z2_pred = self.zf2(obs, actions, tau_hat)
        zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)
        zf1_loss = zf1_loss.mean(axis=2 if self.config['recurrent_policy'] else 1)
        zf2_loss = zf2_loss.mean(axis=2 if self.config['recurrent_policy'] else 1)
        # value_loss = torch.min(zf1_loss, zf2_loss)
        value_loss = zf1_loss + zf2_loss

        # Update priorities in buffer 1
        if self.prioritized_replay:
            if self.config['recurrent_policy']:
                value_loss = value_loss.mean(axis=1)
            td_error = value_loss.cpu().detach().numpy().flatten()
            weights_update = np.abs(td_error) + self.config['priority_epsilon']
            replay_priority_queue.put((inds, weights_update))
            if self.config['recurrent_policy']:
                w_shape = weights.shape[0]
                weights = weights.reshape((w_shape, 1))
            value_loss_1 = zf1_loss * torch.tensor(weights).float().to(self.device)
            value_loss_2 = zf2_loss * torch.tensor(weights).float().to(self.device)
            zf1_loss = value_loss_1.mean(axis=1 if self.config['recurrent_policy'] else 0)
            zf2_loss = value_loss_2.mean(axis=1 if self.config['recurrent_policy'] else 0)

        zf1_loss = zf1_loss.mean()
        zf2_loss = zf2_loss.mean()

        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()

        # ------- Update Policy -------
        with torch.no_grad():
            newtau, new_tau_hat, new_presum_tau = self.get_tau(new_actions.detach())

        z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
        z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
        q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=2 if self.config['recurrent_policy'] else 1, keepdim=True)
        q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=2 if self.config['recurrent_policy'] else 1, keepdim=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = fast_clip_grad_norm(self.policy_net.parameters(), self.clip_norm)
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_zf1.parameters(), self.zf1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.beta) + param.data * self.beta)

        for target_param, param in zip(self.target_zf2.parameters(), self.zf2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.beta) + param.data * self.beta)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.beta) + param.data * self.beta)

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
            logs[4] = value_loss.mean().item()
            logs[5] = time.time() - update_time

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, logs):
        torch.set_num_threads(4)
        time.sleep(2)

        manager = enlighten.get_manager()
        status_format = '{program}{fill}Stage: {stage}{fill} Status {status}'
        algorithm = f"{self.config['model']}-{'P' if self.prioritized_replay else 'N'}"
        status_bar = manager.status_bar(status_format=status_format, color='bold_slategray', program=algorithm, stage=str(self.config['env_stage']), status='Training')
        ticks = manager.counter(total=self.config['num_steps_train'], desc="Training step", unit="ticks", color="red")
        while update_step.value <= self.config['num_steps_train']:
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
        os.system("kill $(ps aux | grep gzserver | grep -v grep | awk '{print $2}')")
        print("Exit learner.")
