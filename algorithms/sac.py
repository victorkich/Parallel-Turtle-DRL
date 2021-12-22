from tensorboardX import SummaryWriter
from models import ActorSAC, Critic, Q
from torch.distributions import Normal
import torch.optim as optim
import torch.nn as nn
import torch


class SAC(object):
    def __init__(self, state_dim, action_dim, config, save_dir):
        self.config = config
        self.update_iteration = config['update_agent_ep']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.device = config['device']
        self.save_dir = save_dir

        self.actor = ActorSAC(state_dim, hidden=config['dense_size']).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])
        self.critic = Critic(state_dim, action_dim, config['dense_size']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])

        self.Q_net = Q(state_dim, action_dim, hidden=config['dense_size']).to(self.device)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=config['q_learning_rate'])
        self.critic_target = Critic(state_dim, action_dim, config['dense_size']).to(self.device)
        self.writer = SummaryWriter(self.save_dir)

        self.critic_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()
        self.num_training = 0

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
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
        batch_mu, batch_log_sigma = self.actor(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self, replay_buffer):
        for it in range(self.update_iteration):
            # Sample replay buffer
            x, u, r, y, d, _ = replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)

            # Compute the target Q value
            target_value = self.critic_target(next_state)
            next_q_value = reward + (1 - done) * self.config['discount_rate'] * target_value
            excepted_value = self.actor(state)
            excepted_Q = self.Q_net(state, action)

            # Get current Q estimate
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(state)
            excepted_new_Q = self.Q_net(state, sample_action)
            next_value = excepted_new_Q - log_prob

            # Compute critic loss
            critic_loss = self.critic_criterion(excepted_value, next_value.detach())  # J_V
            critic_loss = critic_loss.mean()

            # Compute Q loss. Single Q_net this is different from original paper
            Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach())  # J_Q
            Q_loss = Q_loss.mean()

            # Compute actor loss
            log_policy_target = excepted_new_Q - excepted_value
            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()

            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q_loss', Q_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)

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

            # soft update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param * (1 - self.config['tau']) + param * self.config['tau'])

            self.num_training += 1
            return critic_loss

    def save(self):
        torch.save(self.actor.state_dict(), self.save_dir + 'SAC_actor%d.pth' % self.num_training)
        torch.save(self.critic.state_dict(), self.save_dir + 'SAC_critic%d.pth' % self.num_training)
        torch.save(self.Q_net.state_dict(), self.save_dir + 'SAC_q_net%d.pth' % self.num_training)
        print("===================================\nSAC model has been saved...\n===================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.save_dir + 'SAC_actor.pth'))
        self.critic.load_state_dict(torch.load(self.save_dir + 'SAC_critic.pth'))
        self.Q_net.load_state_dict(torch.load(self.save_dir + 'SAC_q_net.pth'))
        print("===================================\nSAC model has been loaded...\n===================================")
