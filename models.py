from utils.utils import hidden_init, TanhNormal, fanin_init
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import abc


class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """
    def __init__(self, num_states, num_actions, hidden_size, v_min, v_max,
                 num_atoms, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): size of the hidden layers
            v_min (float): minimum value for critic
            v_max (float): maximum value for critic
            num_atoms (int): number of atoms in distribution
            init_w:
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_atoms)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_probs(self, state, action):
        return torch.softmax(self.forward(state, action), dim=1)


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int):  action dimension
            hidden_size (int): size of the hidden layer
        """
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.to(device)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def to(self, device):
        super(PolicyNetwork, self).to(device)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action


class PolicyNetwork2(nn.Module):
    """Actor for SAC - return action value given states. """

    def __init__(self, state_size, action_size, device, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyNetwork2, self).__init__()
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

        self.to(device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def to(self, device):
        super(PolicyNetwork2, self).to(device)

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob

    def get_action(self, state, exploitation=False):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        state = torch.FloatTensor(state).to(self.device)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        if not exploitation:
            action = torch.tanh(mu + e * std).cpu()
        else:
            action = torch.tanh(mu).cpu()
        return action


class QuantileMlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            config,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [nn.Linear(last_size, next_size), nn.LayerNorm(next_size) if layer_norm else nn.Identity(), nn.ReLU(inplace=True)]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(nn.Linear(embedding_size, last_size), nn.LayerNorm(last_size) if layer_norm else nn.Identity(), nn.Sigmoid())
        self.merge_fc = nn.Sequential(nn.Linear(last_size, hidden_sizes[-1]), nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(), nn.ReLU(inplace=True))
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).to(config['device'])
        self.to(config['device'])

    def to(self, device):
        super(QuantileMlp, self).to(device)

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output


class Mlp(nn.Module):
    def __init__(self, hidden_sizes, output_size, input_size, config, init_w=3e-3, hidden_activation=F.relu,
                 output_activation=nn.Identity, hidden_init=fanin_init, b_init_value=0.1, layer_norm=False,
                 layer_norm_kwargs=None):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.to(config['device'])

    def to(self, device):
        super(Mlp, self).to(device)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """
        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, config, std=None, init_w=1e-3, **kwargs):
        super().__init__(hidden_sizes, input_size=obs_dim, output_size=action_dim, config=config, init_w=init_w, **kwargs)
        self.config = config
        self.device = config['device']
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert config['v_min'] <= self.log_std <= config['v_max']

        self.to(config['device'])

    def to(self, device):
        super(TanhGaussianPolicy, self).to(device)

    @torch.no_grad()
    def get_action(self, obs_np):
        action, _, _, _, _, _, _, _ = self.forward(obs_np)
        return action

    def forward(self, obs, reparameterize=True, deterministic=False, return_log_prob=False):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, self.config['v_min'], self.config['v_max'])
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std, self.config)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, min_log_std=-20, max_log_std=2, device='cpu'):
        super(ActorSAC, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        self.device = device

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head

    def to(self, device):
        super(ActorSAC, self).to(device)


class ActorDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden, device='cpu'):
        super(ActorDDPG, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, action_dim)
        self.device = device

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def to(self, device):
        super(ActorDDPG, self).to(device)
        

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

