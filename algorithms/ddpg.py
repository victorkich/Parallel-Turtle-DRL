from tensorboardX import SummaryWriter
from models import ActorDDPG, Critic
import torch.nn.functional as F
import torch.optim as optim
import torch


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, config, save_dir):
        self.update_iteration = config['update_agent_ep']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.device = config['device']
        self.save_dir = save_dir

        self.actor = ActorDDPG(state_dim, action_dim, max_action, config['dense_size']).to(self.device)
        self.actor_target = ActorDDPG(state_dim, action_dim, max_action, config['dense_size']).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])

        self.critic = Critic(state_dim, action_dim, config['dense_size']).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, config['dense_size']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])
        self.writer = SummaryWriter(self.save_dir)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer):
        for it in range(self.update_iteration):
            # Sample replay buffer
            x, u, r, y, d, _ = replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

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
            return F.mse_loss(current_Q, target_Q)

    def save(self):
        torch.save(self.actor.state_dict(), self.save_dir + 'DDPG_actor%d.pth' % self.num_actor_update_iteration)
        torch.save(self.critic.state_dict(), self.save_dir + 'DDPG_critic%d.pth' % self.num_critic_update_iteration)
        print("===================================\nDDPG model has been saved...\n===================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.save_dir + 'DDPG_actor.pth'))
        self.critic.load_state_dict(torch.load(self.save_dir + 'DDPG_critic.pth'))
        print("===================================\nDDPG model has been loaded...\n===================================")
