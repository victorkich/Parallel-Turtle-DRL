from utils.utils import empty_torch_queue
from models import Critic
import torch.nn.functional as F
import torch.optim as optim
import torch
import queue
import time


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

    def _update_step(self, replay_buffer, update_step, logs):
        update_time = time.time()

        # Sample replay buffer
        x, u, r, y, d, gamma, weights, inds = replay_buffer
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1-d).to(self.device).unsqueeze(1)
        reward = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        # weights = torch.FloatTensor(weights).to(self.device)
        # inds = torch.FloatTensor(inds).flatten().to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
            logs[3] = critic_loss
            logs[4] = actor_loss
            logs[5] = time.time() - update_time

    def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
        torch.set_num_threads(4)
        while global_episode.value <= self.config['num_agents'] * self.config['num_episodes']:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            self._update_step(batch, update_step, logs)
            with update_step.get_lock():
                update_step.value += 1

            if update_step.value % 10000 == 0:
                print("Training step ", update_step.value)

        with training_on.get_lock():
            training_on.value = 0

        empty_torch_queue(self.learner_w_queue)
        empty_torch_queue(replay_priority_queue)
        print("Exit learner.")