import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import numpy as np

'''
Elements of RNN logic adapted from https://github.com/vwxyzjn/ppo-implementation-details,
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#Schulman2017,
and https://github.com/lucaslingle/pytorch_rl2.
'''


class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.).float()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class ControllerLatent(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, latent_type=nn.GRU):
        # input_dim: number of input features
        # hidden_dim: number of hidden units
        super().__init__()
        self._hidden_dim = hidden_dim
        self.device = device
        self.latentRNN = latent_type(input_dim, hidden_dim, device=self.device)
        self._latent_type = type(self.latentRNN)

    def forward(self, x, rnn_state=None):
        # updates hidden state w new inputs

        batched = len(x.shape) == 4
        if batched:
            x = x.permute((2, 0, 1, 3))

        output = []
        if self._latent_type is nn.LSTM:
            if rnn_state is None:
                hidden = torch.zeros((1, self._hidden_dim), device=self.device)
                cell = torch.zeros((1, self._hidden_dim), device=self.device)
                rnn_state = (hidden, cell)

            for obs in x:
                out, rnn_state = self.latentRNN(obs, (rnn_state[0], rnn_state[1]))
                output.append(out)
        else:
            if rnn_state is None:
                rnn_state = torch.zeros((1, self._hidden_dim), device=self.device)

            for obs in x:
                out, rnn_state = self.latentRNN(obs, rnn_state)
                output.append(out)

        output = torch.cat(output)
        if batched:
            output = output.permute((1,0,2))
        return output, rnn_state

    @property
    def latent_type(self):
        return self._latent_type


class Policy(nn.Module):
    # outputs action given controller's current latent state
    def __init__(self, hidden_dim, n_actions, n_policy_hidden, device):
        # n_actions: number of output control actions
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, n_policy_hidden, device=device)
        self.fc2 = nn.Linear(n_policy_hidden, n_actions, device=device)

        self.act_dim = n_actions
        self.device = device

    def forward(self, rnn_output, sample_action=True):
        y = self.fc1(rnn_output)
        y = F.relu(y)
        y = self.fc2(y)
        probs = torch.sigmoid(y)

        if sample_action:
            assert rnn_output.shape[0] == 1
            action_sampler = Bernoulli(probs=probs.detach())
            actions = action_sampler.sample()

            # rand_vec = torch.rand((1, self.act_dim), device=self.device)
            # actions = Threshold.apply(probs - rand_vec)
        else:
            actions = None

        if actions is None:
            # print("action prob: ", probs.detach().numpy(), y.detach().numpy(), "None")
            pass
        else:
            log_action_p = torch.sum(actions * torch.log(probs), dim=-1) + torch.sum(
                (1. - actions) * torch.log(1. - probs), dim=-1)
            action_p = log_action_p.exp()
            # print("action prob: ", probs.detach().numpy(), y.detach().numpy(), actions.detach().numpy())
            # print("action probs2: ", action_p)

        return actions, probs


class Controller(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, n_policy_hidden, device):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._latent_module = ControllerLatent(input_dim, hidden_dim, device)
        self._actor_module = Policy(hidden_dim, n_actions, n_policy_hidden, device)
        self._n_policy_hidden = n_policy_hidden
        self._critic_module = Value(hidden_dim, hidden_dim, device)

    def forward(self, x, rnn_state=None, action=None):
        batched = rnn_state.shape[0] > 1
        if batched:
            rnn_state = rnn_state.unsqueeze(0)

        # takes in last hidden state and feedback x from last action, and returns updated hidden and new action(model)
        x = x.unsqueeze(0)
        rnn_output, rnn_state = self._latent_module(x, rnn_state)

        if batched:
            rnn_state = rnn_state.squeeze(0)

        if action is None:
            action, action_p = self._actor_module(rnn_output)
        else:
            _, action_p = self._actor_module(rnn_output, sample_action=False)

        # keep dim in sum only for unbatched inputs
        log_action_p = torch.sum(action*torch.log(action_p), dim=-1, keepdim=not batched) + torch.sum((1.-action)*torch.log(1.-action_p), dim=-1, keepdim=not batched)
        value = self._critic_module(rnn_output.detach())

        if batched:
            value = value.squeeze(-1)

        return action, log_action_p, value, rnn_state

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def latent_module(self):
        return self._latent_module

    @property
    def actor_module(self):
        return self._actor_module

    @property
    def critic_module(self):
        return self._critic_module

    @property
    def n_policy_hidden(self):
        return self._n_policy_hidden

    @property
    def latent_type(self):
        return self.latent_module.latent_type


class Critic(nn.Module):
    def __init__(self, hidden_dim, n_actions, n_critic_hidden, device):
        # hidden_dim: number of hidden units in LSTM controller
        # n_actions: number of output control actions
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim + n_actions, n_critic_hidden, device=device)
        self.fc2 = nn.Linear(n_critic_hidden, 1, device=device)

    def forward(self, hidden, gate):
        # hidden: LSTM hidden state
        # gate: a gating action
        x = torch.cat((hidden, gate), dim=1)
        y = self.fc1(x)
        y = F.relu(y)
        Q_val = self.fc2(y)
        return Q_val


class Value(nn.Module):
    def __init__(self, hidden_dim, n_value_hidden, device):
        # hidden_dim: number of hidden units in LSTM controller
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, n_value_hidden, device=device)
        self.fc2 = nn.Linear(n_value_hidden, 1, device=device)

    def forward(self, hidden):
        # hidden: LSTM hidden state
        y = self.fc1(hidden)
        y = F.relu(y)
        V = self.fc2(y)
        return V


class Target(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()

        self._n_input = input_dim
        self._n_hidden = hidden_dim
        self._n_output = output_dim
        self.fc1 = nn.Linear(self.n_input, self.n_hidden, device=device)
        self.fc2 = nn.Linear(self.n_hidden, self.n_output, device=device)
        self._loss = nn.MSELoss()

    @property
    def n_input(self):
        return self._n_input

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_output(self):
        return self._n_output

    def forward(self, x, gating_action):
        x = self.fc1(x)
        x = F.relu(x)
        x = x * gating_action.detach()
        x = self.fc2(x)
        return x

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    @torch.no_grad()
    def reset_weights(self):
        std1 = 10*np.random.rand()
        std2 = np.random.rand()
        # weight vectors chosen randomly w uniformly distributed direction and normally distributed magnitude
        for param in self.fc1.parameters():
            nn.init.normal_(param, std=std1)

        for param in self.fc2.parameters():
            nn.init.normal_(param, std=std2)


class TrainDataNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, device):
        super().__init__()

        self._n_input = n_input
        self._n_hidden = n_hidden
        self._n_output = n_output
        self.fc1 = nn.Linear(self._n_input, self._n_hidden, device=device)
        self.fc2 = nn.Linear(self._n_hidden, self._n_output, device=device)

    @torch.inference_mode()
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def reset_weights(self, std=1):
        for param in self.parameters():
            nn.init.normal_(param, std=std)
