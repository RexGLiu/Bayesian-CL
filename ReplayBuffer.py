import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, latent_dim, act_dim):
        self.size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.pos = 0
        self.full = False

        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.hidden = np.zeros((self.size, self.latent_dim), dtype=np.float32)
        self.cell = np.zeros((self.size, self.latent_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.bool_)

    def add(self, obs, hidden, cell, action, reward, done):
        # inserts single data sample

        self.obs[self.pos] = obs
        self.hidden[self.pos] = hidden
        self.cell[self.pos] = cell
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        if self.pos + 1 == self.size:  # buffer full
            self.full = True
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size=1, sample_len=80):
        if self.full:
            indices = np.random.choice(self.size, batch_size)
            end_indices = np.minimum(indices + sample_len, self.size)
            end_indices[(indices <= self.pos)*(self.pos < end_indices)] = self.pos
        else:
            indices = np.random.choice(self.pos, batch_size)
            end_indices = np.minimum(indices + sample_len, self.pos)

        return self.obs[indices:end_indices], self.hidden[indices], self.cell[indices], \
               self.actions[indices:end_indices], self.rewards[indices:end_indices], self.dones[indices:end_indices]
