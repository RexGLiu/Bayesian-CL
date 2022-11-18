import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        # input_dim: number of input features
        # hidden_dim: number of hidden units
        # n_actions: number of output control actions
        super().__init__()

        self._hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden=None, cell=None):
        if hidden is None:
            hidden = torch.zeros(self._hidden_dim)
        if cell is None:
            cell = torch.zeros(self._hidden_dim)

        x, (hidden, cell) = self.LSTM(x, (hidden, cell))
        y = self.hidden_to_policy(x)
        return x, (y, hidden, cell)

    def hidden_to_policy(self, hidden):
        y = self.fc(hidden)
        y = F.sigmoid(y)
        return y

    @staticmethod
    def sample(p, thresh=0.0):
        if type(p) is torch.Tensor:
            p = p.numpy()
        if thresh == 0.0:
            return np.random.binomial(1, p=p)
        else:
            return (p > thresh).astype(np.int_)

    @property
    def hidden_dim(self):
        return self._hidden_dim


class Critic(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        # hidden_dim: number of hidden units in LSTM controller
        # n_actions: number of output control actions
        super().__init__()

        self.fc = nn.Linear(hidden_dim + n_actions, 1)

    def forward(self, hidden, gate):
        # hidden: LSTM hidden state
        # gate: a gating action
        x = torch.cat((hidden, gate))
        Q_val = self.fc(x)
        return Q_val


class Value(nn.Module):
    def __init__(self, hidden_dim):
        # hidden_dim: number of hidden units in LSTM controller
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, hidden):
        # hidden: LSTM hidden state
        V = self.fc(hidden)
        return V


class Target(nn.Module):
    def __init__(self):
        super().__init__()

        self._n_input = 2
        self._n_hidden = 10
        self._n_output = 2
        self.fc1 = nn.Linear(self.n_input, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_output)

        # input_dim_c = n_input_t + n_hidden_t + 2*n_output_t + 1
        # n_hidden_c = 48
        # self.controller = Controller(input_dim_c,n_hidden_c,n_hidden_t)

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
        # try:
        #     assert gating_action.requires_grad is False
        # except:
        #     print("Error: must stop gradients in gating actions.")
        #     raise

        x = self.fc1(x)
        x = F.relu(x)
        x = x * gating_action
        x = self.fc2(x)

        return x

    @staticmethod
    def loss(y_hat, y):
        return nn.MSELoss(y_hat, y)

    @torch.no_grad()
    def reset_weights(self):
        for param in self.parameters():
            nn.init.xavier_normal_(param)


class TrainNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        self._n_input = n_input
        self._n_hidden = n_hidden
        self._n_output = n_output
        self.fc1 = nn.Linear(self.n_input, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def reset_weights(self, gain=1):
        for param in self.parameters():
            nn.init.xavier_normal_(param, gain=gain)
