import gym
from gym import spaces
import numpy as np

import torch


class TargetNetEnv(gym.Env):
    # interaction with the target network framed as an RL env

    def __init__(self, target_net, device):
        self.target_net = target_net
        self.target_net.requires_grad_(False)
        self.target_net.eval()

        self.device = device

        self._n_input = target_net.n_input
        self._n_output = target_net.n_output
        self._n_nodes = target_net.n_hidden

        self.observation_space = spaces.Box(low=np.full(self._n_output, -np.inf), high=np.full(self._n_output, np.inf),
                                            dtype=np.double)
        self.action_space = spaces.MultiDiscrete(np.ones(self._n_nodes, dtype=int)*2, dtype=int)
        self.reward_range = (0, float("inf"))
        self.render_mode = None

    def reset(self):
        # randomly reinitialises target network weights
        self.target_net.reset_weights()

    def step(self, inputs):
        try:
            assert len(inputs) == 3
        except AssertionError:
            print("TargetNetEnv.step(): Argument 'inputs' must be tuple of form (x, y, gating_action).")
            raise

        x, y, gating_action = inputs
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y).float().to(self.device)
        if type(gating_action) is np.ndarray:
            gating_action = torch.from_numpy(gating_action)

        with torch.no_grad():
            y_hat = self.target_net(x, gating_action)
            loss = self.target_net.loss(y_hat, y)

        return y_hat, loss

    @property
    def act_dim(self):
        return self._n_nodes

    @property
    def input_dim(self):
        return self._n_input

    @property
    def output_dim(self):
        return self._n_output

    def gen_training_data(self, x, gating):
        return self.target_net(x, gating)
