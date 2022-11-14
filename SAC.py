import numpy as np
from .TaskBuffer import TaskBuffer as Buffer
from .model import TrainNet, Critic, Value

import torch
from torch.optim import Adam


class SAC:
    def __init__(self, env, actor, n_buffer_tasks=1E4, n_train_steps=2E7, train_start=5E3,
                 set_min=100, set_max=1000, batch_size = 2, latent_burn_in=True, burn_in_frac=0.5, sample_thresh=0.5,
                 lr=3E-4, smoothing_coeff=0.005, optimizer=Adam, data_net_hidden = 8):
        self.env = env

        self.act_dim = env.n_actions
        self.input_dim = env.input_dim
        self.output_dim = env.output_dim
        self.obs_dim = self.input_dim + 2*self.output_dim

        self.train_start = int(train_start)
        self.n_train_steps = int(n_train_steps)

        self.data_net = TrainNet(self.input_dim, self.output_dim, data_net_hidden)

        self.actor = actor
        self.critic1 = Critic(actor.hidden_size, self.act_dim)
        self.critic2 = Critic(actor.hidden_size, self.act_dim)
        self.value = Value(actor.hidden_size)
        self.target_value = Value(actor.hidden_size)

        self.n_buffer_tasks = int(n_buffer_tasks)
        self.buffer = Buffer(self.n_buffer_tasks, self.set_max, self.input_dim, self.output_dim)
        self.latent_burn_in = latent_burn_in
        self.burn_in_frac = burn_in_frac

        self.sample_thresh = sample_thresh

        self.set_min = set_min
        self.set_max = set_max
        self.batch_size = batch_size

        self.lr = lr
        self.smoothing_coeff = smoothing_coeff
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optimizer(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optimizer(self.critic2.parameters(), lr=self.lr)
        self.value_optimizer = optimizer(self.value.parameters(), lr=self.lr)


    def train(self):

        # initial steps will only populate task buffer with training sets
        for step in range(self.train_start):
            train_x, train_y, set_size = self._generate_training_data()
            self.buffer.add(train_x, train_y, set_size)

        # training loop
        for step in range(self.train_start + 1, self.n_train_steps):
            train_x, train_y, set_size = self._generate_training_data()
            self.buffer.add(train_x, train_y, set_size)

            # retrieve training set to train on
            x_batch, y_batch, batch_set_sizes = self.buffer.sample(self.batch_size)

            # train on retrieved batches
            for ii in range(self.batch_size):
                set_size = batch_set_sizes[ii]
                train_x = x_batch[ii, :set_size]
                train_y = y_batch[ii, :set_size]

                idx_permutation = np.random.permutation(set_size)
                train_x = train_x[idx_permutation]
                train_y = train_y[idx_permutation]

                n_burn_in_steps = 0
                if self.latent_burn_in:
                    n_burn_in_steps = round(set_size * burn_in_frac)
                    n_burn_in_steps = np.random.choice(n_burn_in_steps)

                # first action is arbitrary
                input = np.zeros(self.obs_dim + self.act_dim + 1)
                hidden = None
                cell = None

                # burn in steps to initialize actor hidden state
                for jj in range(n_burn_in_steps):
                    _, (action_p, hidden, cell) = self.actor.forward(input, hidden, cell)
                    action = self.actor.sample(p=action_p, thresh=self.sample_thresh)

                    x = train_x[jj]
                    y_target = train_y[jj]
                    y, loss = self.env.step((x, y_target, action))
                    y = y.numpy()
                    loss = loss.numpy()

                    input = np.concatenate((x, y, y_target, loss, action), axis=1)

                # training steps
                for jj in range(n_burn_in_steps, set_size):
                    _, (action_p, hidden, cell) = self.actor.forward(input, hidden, cell)
                    action = self.actor.sample(p=action_p, thresh=self.sample_thresh)

                    x = train_x[jj]
                    y_target = train_y[jj]
                    y, loss = self.env.step((x, y_target, action))
                    y = y.numpy()
                    loss = loss.numpy()

                    input = np.concatenate((x, y, y_target, loss, action), axis=1)

                    obs = torch.from_numpy(obs)
                    hidden = torch.from_numpy(hidden)
                    cell = torch.from_numpy(cell)
                    action = torch.from_numpy(action)
                    loss = torch.from_numpy(loss)



    def _evaluate(self):
        pass

    def _generate_training_data(self):
        set_size = np.random.randint(self.set_min, self.set_max + 1)
        train_x = np.zeros(set_size, self.input_dim, dtype=np.float32)
        train_y = np.zeros(set_size, self.output_dim, dtype=np.float32)

        # set noise level for output
        max_std = 3
        y_noise_std = max_std * np.random.uniform()  # uniformly sample Gaussian noise variance

        # randomly reset data-generating network weights
        gain = np.random.uniform(1, 5)
        self.data_net.reset_weights(gain)

        x_std = 10
        with torch.inference_mode():
            for ii in range(set_size):
                x_sample = x_std*torch.randn(size=(1, self.input_dim))
                y_sample = self.data_net(x_sample)

                # add noise to train_y
                y_noise = y_noise_std * torch.randn(size=(1, self.output_dim))
                y_sample += y_noise

                # store dataset
                train_x[ii] = x_sample.numpy()
                train_y[ii] = y_sample.numpy()

        return train_x, train_y, set_size