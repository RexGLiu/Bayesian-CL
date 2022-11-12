import numpy as np
from .ReplayBuffer import ReplayBuffer
from .model import TrainNet, Critic, Value

import torch
from torch.optim import Adam


class SAC:
    def __init__(self, env, actor, buffer_size=1E6, n_train_steps=2E7, train_start=5E3,
                 latent_burn_in=40, ep_min=100, ep_max=1000, sample_thresh=0.5,
                 lr=3E-4, smoothing_coeff=0.005, optimizer=Adam):
        self.env = env

        self.act_dim = env.n_actions
        self.input_dim = env.input_dim
        self.output_dim = env.output_dim
        self.obs_dim = self.input_dim + 2*self.output_dim

        self.train_start = int(train_start)
        self.n_train_steps = int(n_train_steps)

        data_net_hidden = 8
        self.data_net = TrainNet(self.input_dim, self.output_dim, data_net_hidden)

        self.actor = actor
        self.critic1 = Critic(actor.hidden_size, self.act_dim)
        self.critic2 = Critic(actor.hidden_size, self.act_dim)
        self.value = Value(actor.hidden_size)
        self.target_value = Value(actor.hidden_size)

        self.buffer_size = int(buffer_size)
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, actor.hidden_size, self.act_dim)
        self.latent_burn_in = latent_burn_in

        self.sample_thresh = sample_thresh

        self.ep_min = ep_min
        self.ep_max = ep_max
        self.ep_len = 0
        self.train_x = None
        self.train_y = None

        self.lr = lr
        self.smoothing_coeff = smoothing_coeff
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optimizer(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optimizer(self.critic2.parameters(), lr=self.lr)
        self.value_optimizer = optimizer(self.value.parameters(), lr=self.lr)


    def train(self):
        env_step = 0
        for step in range(self.train_start):
            if env_step == 0:
                # randomize target network and generate new training data
                self._generate_training_data()
                self.env.reset()

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM and will not be stored in replay buffer
                obs = np.zeros(self.obs_dim + self.act_dim + 1)
                hidden = None
                cell = None

            _, (action_p, hidden, cell) = self.actor.forward(obs, hidden, cell)
            action = self.actor.sample(p=action_p, thresh=self.sample_thresh)

            x = self.train_x[env_step]
            y_target = self.train_y[env_step]
            y, loss = self.env.step((x, y_target, action))
            y = y.numpy()
            loss = loss.numpy()

            obs = np.concatenate((x, y, y_target), axis=1)

            # change environment and target func controller must optimize
            env_step += 1
            done = (env_step == self.ep_len)
            if done:
                env_step = 0

            self.buffer.add(obs, hidden, cell, action, loss, done)

        for step in range(self.train_start+1, self.n_train_steps):
            if env_step == 0:
                # randomize target network and generate new training data
                self._generate_training_data()
                self.env.reset()

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM and will not be stored in replay buffer
                obs = np.zeros(self.obs_dim + self.act_dim + 1)
                hidden = np.zeros(self.actor.hidden_size)
                cell = np.zeros(self.actor.hidden_size)

            _, (action_p, hidden2, cell2) = self.actor.forward(obs, hidden, cell)
            action = self.actor.sample(p=action_p, thresh=self.sample_thresh)

            # get "training data" for target network
            x = self.train_x[env_step]
            y_target = self.train_y[env_step]
            y, loss = self.env.step((x, y_target, action))
            y = y.numpy()
            loss = loss.numpy()

            obs = np.concatenate((x, y, y_target), axis=1)

            # if training set complete, change environment and target func controller must optimize
            env_step += 1
            done = (env_step == self.ep_len)
            if done:
                env_step = 0

            self.buffer.add(obs, hidden, cell, action, loss, done)
            hidden = hidden2
            cell = cell2

            self._grad_update()

    def _grad_update(self):
        # Compute gradients
        obs, hidden, cell, action, rew, done = self.buffer.sample()
        obs = torch.from_numpy(obs)
        hidden = torch.from_numpy(hidden)
        cell = torch.from_numpy(cell)
        action = torch.from_numpy(action)
        rew = torch.from_numpy(rew)

        # burn in of LSTM hidden states
        for ii in range(self.latent_burn_in):
            _, (_, hidden, cell) = self.actor.forward(obs, hidden, cell)

        # asd

    def _evaluate(self):
        pass

    def _generate_training_data(self):
        self.ep_len = np.random.randint(self.ep_min, self.ep_max + 1)
        self.train_x = np.zeros(self.ep_len, self.input_dim, dtype=np.float32)
        self.train_y = np.zeros(self.ep_len, self.output_dim, dtype=np.float32)

        # set noise level for output
        max_std = 3
        y_noise_std = max_std * np.random.uniform()  # uniformly sample Gaussian noise variance

        # randomly reset data-generating network weights
        gain = np.random.uniform(1, 5)
        self.data_net.reset_weights(gain)

        x_std = 10
        with torch.inference_mode():
            for ii in range(self.ep_len):
                x_sample = x_std*torch.randn(size=(1, self.input_dim))
                y_sample = self.data_net(x_sample)

                # add noise to train_y
                y_noise = y_noise_std * torch.randn(size=(1, self.output_dim))
                y_sample += y_noise

                # store dataset
                self.train_x[ii] = x_sample.numpy()
                self.train_y[ii] = y_sample.numpy()
