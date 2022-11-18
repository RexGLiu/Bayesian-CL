import numpy as np
from .ReplayBuffer import ReplayBuffer
from .model import TrainNet, Critic, Value

import torch
import torch.nn as nn
from torch.optim import Adam


class SAC:
    def __init__(self, env, actor, buffer_size=1E6, n_train_steps=2E7, train_start=5E3,
                 latent_burn_in=40, ep_min=100, ep_max=1000, sample_thresh=0.5, action_penalty_coeff = 0.5,
                 lr=3E-4, ent_coeff = 1., smoothing_coeff=0.005, gamma=0.999, optimizer=Adam):
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
        self.value_target = Value(actor.hidden_size)

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

        self.gamma = gamma
        self.ent_coeff = ent_coeff

        self.action_pen = action_penalty_coeff


    def train(self):
        env_step = 0
        for step in range(self.train_start):
            if env_step == 0:
                # randomize target network and generate new training data
                self._generate_training_data()
                self.env.reset()

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM and will not be stored in replay buffer
                inputs = torch.zeros(self.obs_dim + self.act_dim + 3)
                hidden = None
                cell = None

            _, (action_p, hidden, cell) = self.actor.forward(inputs, hidden, cell)
            action = self.actor.sample(p=action_p, thresh=self.sample_thresh)
            action_ent = self._action_ent(action, action_p)
            action_ent = action_ent.numpy()
            action_size = torch.sum(action)
            action_size = action_size.numpy()

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

            self.buffer.add(obs, hidden.numpy(), cell.numpy(), action, loss, done)
            inputs = np.concatenate((obs, action, loss, action_size, action_ent), axis=1)
            inputs = torch.from_numpy(inputs)

        for step in range(self.train_start+1, self.n_train_steps):
            if env_step == 0:
                # randomize target network and generate new training data
                self._generate_training_data()
                self.env.reset()

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM and will not be stored in replay buffer
                inputs = torch.zeros(self.obs_dim + self.act_dim + 3)
                hidden = None
                cell = None

            _, (action_p, hidden, cell) = self.actor.forward(inputs, hidden, cell)
            action = self.actor.sample(p=action_p, thresh=self.sample_thresh)
            action_ent = self._action_ent(action, action_p)
            action_ent = action_ent.numpy()
            action_size = torch.sum(action)
            action_size = action_size.numpy()

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

            self.buffer.add(obs, hidden.numpy(), cell.numpy(), action, loss, done)
            inputs = np.concatenate((obs, action, loss, action_size, action_ent), axis=1)
            inputs = torch.from_numpy(inputs)

            self._grad_update()

    @staticmethod
    def _action_ent(action, action_p):
        if type(action) is torch.Tensor:
            return -torch.sum(torch.log(action*action_p)) - torch.sum(torch.log((1-action) * (1-action_p)))
        else:
            return -np.sum(np.log(action * action_p)) - np.sum(np.log((1-action) * (1-action_p) ))

    def _reward(self, R, action_len, entropy):
        return R - self.action_pen * action_len * action_len + self.ent_coeff * entropy

    def _grad_update(self):
        # Compute gradients
        obs_array, hidden, cell, action_array, loss_array, done_array = self.buffer.sample()
        hidden = torch.from_numpy(hidden)
        cell = torch.from_numpy(cell)
        action_p = self.actor.hidden_to_policy(hidden)  # get policy probs from LSTM's hidden state

        sample_len = obs_array.size
        if sample_len > self.latent_burn_in:  # only perform burn-in if buffer returned sample length that's long enough
            burn_in_steps = self.latent_burn_in
        else:
            burn_in_steps = 0

        # burn in of LSTM hidden states
        for ii in range(burn_in_steps):
            obs = torch.from_numpy(obs_array[ii])
            action = torch.from_numpy(action_array[ii])
            loss = torch.from_numpy(loss_array[ii])

            # Note: action_p will change over course of training. To avoid staleness, we recompute entropy term.
            action_size = torch.sum(action)
            action_ent = self._action_ent(action, action_p)
            action_ent = action_ent.numpy()

            inputs = np.concatenate((obs, action, loss, action_size, action_ent), axis=1)
            inputs = torch.from_numpy(inputs)
            _, (action_p, hidden, cell) = self.actor.forward(inputs, hidden, cell)

            if done_array[ii]:  # if starting new task, reinitialize latents to 0 and perform initial LSTM dummy step
                inputs = torch.zeros(self.obs_dim + self.act_dim + 3)
                hidden = None
                cell = None
                _, (action_p, hidden, cell) = self.actor.forward(inputs, hidden, cell)

        # gradient-update loop
        value_loss = torch.tensor(0., requires_grad=False)
        Q1_loss = torch.tensor(0., requires_grad=False)
        Q2_loss = torch.tensor(0., requires_grad=False)
        for ii in range(burn_in_steps, sample_len):
            obs = torch.from_numpy(obs_array[ii])
            action = torch.from_numpy(action_array[ii])
            loss = torch.from_numpy(loss_array[ii])

            # Note: action_p will change over course of training. To avoid staleness, we recompute entropy term.
            action_size = torch.sum(action)
            action_ent = self._action_ent(action, action_p)
            action_ent = action_ent.numpy()

            inputs = np.concatenate((obs, action, loss, action_size, action_ent), axis=1)
            inputs = torch.from_numpy(inputs)

            _, (action_p, hidden2, cell2) = self.actor.forward(inputs, hidden, cell)

            # detach hidden state tensor for critic networks so that grads don't propagate through LSTM
            critic_hidden = hidden.detach()
            critic_hidden.requires_grad = True

            critic_hidden2 = hidden2.detach()
            critic_hidden2.requires_grad = True

            # compute target for V
            with torch.no_grad():
                action_sample = self.actor.sample(p=action_p, thresh=self.sample_thresh)
                action_ent = self._action_ent(action_sample, action_p)
                Q1 = self.critic1(critic_hidden, action_sample)
                Q2 = self.critic2(critic_hidden, action_sample)
                Q = torch.minimum(Q1, Q2)
                target_value = Q + action_ent

            # track losses of V across trajectory (note: we accumulate losses then backprop at end, like a minibatch)
            value = self.value(critic_hidden2)
            value_loss = value_loss + nn.MSELoss(value, target_value)

            # compute target for Q
            with torch.no_grad():
                Q_target = -loss + self.action_pen * action_size + self.value(hidden2)

            # track losses of Q across trajectory (note: we accumulate losses then backprop at end, like a minibatch)
            Q = self.critic1(critic_hidden, action)
            Q1_loss = Q1_loss + nn.MSELoss(Q, Q_target)

            Q = self.critic2(critic_hidden, action)
            Q2_loss = Q2_loss + nn.MSELoss(Q, Q_target)

            # update policy via reparametrization trick (like in SAC paper)
            # policy network : K unit rand vars + state --> coords in K-dim unit cube
            # backprop via straight-through grads
            # not Gumbel bc this treats every dim independent; otherwise need 2^K dim output for all possible combos


        # backprop and update weights


        # update critic targets via moving average









            if done_array[ii]:  # if starting new task, reinitialize latents to 0 and perform initial LSTM dummy step
                inputs = torch.zeros(self.obs_dim + self.act_dim + 1)
                hidden = None
                cell = None
                _, (_, hidden, cell) = self.actor.forward(inputs, hidden, cell)

    def _evaluate(self):
        # used for evaluating training progress
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