import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import queue
from utils.utils import OUNoise, empty_torch_queue
from torch.distributions import MultivariateNormal
from models import ValueNetwork
from utils.l2_projection import _l2_project


class LearnerDSAC(object):
    """Policy and value network update routine. """

    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        self.config = config
        action_low = [-1.5, -0.1]
        action_high = [1.5, 0.12]
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
        self.alpha = 1
        self.log_alpha = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True).to(config['device']))
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=config['actor_learning_rate'])
        self._action_prior = config['action_prior']
        self.target_entropy = -config['action_dim']
        self.action_size = config['action_dim']

        # Noise process
        self.ou_noise = OUNoise(dim=config['action_dim'], low=action_low, high=action_high)

        # Value 1 nets
        self.value_net_1 = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.target_value_net_1 = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], self.v_min, self.v_max, self.num_atoms, device=self.device)
        for target_param, param in zip(self.target_value_net_1.parameters(), self.value_net_1.parameters()):
            target_param.data.copy_(param.data)
        
        #value 2 nets
        self.value_net_2 = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.target_value_net_2 = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], self.v_min, self.v_max, self.num_atoms, device=self.device)
        for target_param, param in zip(self.target_value_net_2.parameters(), self.value_net_2.parameters()):
            target_param.data.copy_(param.data)
        
        #policy nets
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.value_optimizer_1 = optim.Adam(self.value_net_1.parameters(), lr=value_lr)
        self.value_optimizer_2 = optim.Adam(self.value_net_2.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.BCELoss(reduction='none')

    def _update_step(self, batch, replay_priority_queue, update_step, logs):
        update_time = time.time()

        state, action, reward, next_state, done, gamma, weights, inds = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        # ------- Update critic -------

        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.target_policy_net.evaluate(next_state)

        # Predict Z distribution with target value network
        target_value_1 = self.target_value_net_1.get_probs(next_state, next_action.detach())
        target_value_2 = self.target_value_net_2.get_probs(next_state, next_action.detach())

        # take the mean of both critics for updating
        target_value_next = torch.min(target_value_1, target_value_2)

        # Get projected distribution
        target_z_projected = _l2_project(next_distr_v=target_value_next, rewards_v=reward, dones_mask_t=done,
                                         gamma=self.gamma ** 5, n_atoms=self.num_atoms, v_min=self.v_min,
                                         v_max=self.v_max, delta_z=self.delta_z)
        target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device)

        if not self.config['fixed_alpha']:
            # Compute Q targets for current states (y_i)
            target_z_projected = target_z_projected - self.alpha * log_pis_next.mean().squeeze(0)
        else:
            target_z_projected = target_z_projected - self.config['fixed_alpha'] * log_pis_next.mean().squeeze(0)

        critic_value_1 = self.value_net_1.get_probs(state, action)
        critic_value_1 = critic_value_1.to(self.device)

        value_loss_1 = self.value_criterion(critic_value_1, target_z_projected.detach())
        value_loss_1 = value_loss_1.mean(axis=1)

        critic_value_2 = self.value_net_2.get_probs(state, action)
        critic_value_2 = critic_value_2.to(self.device)

        value_loss_2 = self.value_criterion(critic_value_2, target_z_projected.detach())
        value_loss_2 = value_loss_2.mean(axis=1)

        # Update priorities in buffer 1
        value_loss = torch.min(value_loss_1, value_loss_2)
        td_error = value_loss.cpu().detach().numpy().flatten()

        if self.prioritized_replay:
            weights_update = np.abs(td_error) + self.config['priority_epsilon']
            replay_priority_queue.put((inds, weights_update))
            value_loss_1 = value_loss_1 * torch.tensor(weights).float().to(self.device)
            value_loss_2 = value_loss_2 * torch.tensor(weights).float().to(self.device)

        # Update step 1
        value_loss_1 = value_loss_1.mean()
        self.value_optimizer_1.zero_grad()
        value_loss_1.backward()
        self.value_optimizer_1.step()

        # Update step
        value_loss_2 = value_loss_2.mean()
        self.value_optimizer_2.zero_grad()
        value_loss_2.backward()
        self.value_optimizer_2.step()

        # -------- Update actor -----------
        actions_pred, log_pis = self.policy_net.evaluate(state)
        if not self.config['fixed_alpha']:
            alpha = torch.exp(self.log_alpha)
            # Compute alpha loss
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = alpha
            # Compute actor loss
            if self._action_prior == "normal":
                policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size).to(self.config['device']),
                                                  scale_tril=torch.eye(self.action_size).to(self.config['device']))
                policy_prior_log_probs = policy_prior.log_prob(actions_pred)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0

            actor_loss_1 = (alpha * log_pis.mean().squeeze(0) - self.value_net_1.get_probs(state, actions_pred.squeeze(0)) - policy_prior_log_probs.mean()).mean()
            actor_loss_2 = (alpha * log_pis.mean().squeeze(0) - self.value_net_2.get_probs(state, actions_pred.squeeze(0)) - policy_prior_log_probs.mean()).mean()
            policy_loss = torch.min(actor_loss_1, actor_loss_2)
        else:
            if self._action_prior == "normal":
                policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size).to(self.config['device']),
                                                  scale_tril=torch.eye(self.action_size).to(self.config['device']))
                policy_prior_log_probs = policy_prior.log_prob(actions_pred)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0

            actor_loss_1 = (self.config['fixed_alpha'] * log_pis.mean().squeeze(0) -
                            self.value_net_1.get_probs(state, actions_pred.squeeze(0)) - policy_prior_log_probs.mean()).mean()
            actor_loss_2 = (self.config['fixed_alpha'] * log_pis.mean().squeeze(0) -
                            self.value_net_2.get_probs(state, actions_pred.squeeze(0)) - policy_prior_log_probs.mean()).mean()
            policy_loss = torch.min(actor_loss_1, actor_loss_2)

        policy_loss = policy_loss * torch.from_numpy(self.value_net_1.z_atoms).float().to(self.device)
        print(policy_loss)
        policy_loss = torch.sum(policy_loss.squeeze(0))
        policy_loss = policy_loss.mean()
        print(policy_loss.item())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net_1.parameters(), self.value_net_1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_value_net_2.parameters(), self.value_net_2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[3] = policy_loss.item()
            logs[4] = value_loss.item()
            logs[5] = time.time() - update_time

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
        torch.set_num_threads(4)
        while global_episode.value <= self.config['num_agents'] * self.config['num_episodes']:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
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
