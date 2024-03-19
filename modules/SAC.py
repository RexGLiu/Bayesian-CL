import numpy as np
from .ReplayBuffer import ReplayBuffer
from .model import TrainDataNet, Critic, Value

import torch
import torch.nn as nn
from torch.optim import Adam


def PolyakUpdate(params, target_params, alpha):
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1-alpha)
            target_param.data.add_(param.data, alpha=alpha)


class SAC:
    def __init__(self, env, controller, device, buffer_size=1E6, n_train_steps=2E7, train_start=5E3,
                 latent_burn_in=0, ep_min=100, ep_max=1000, sample_thresh=0.5, action_penalty_coeff=0.001, ent_coeff=0.001,
                 lr_value=1E-3, lr_critic=1E-3, lr_actor=1E-4, n_minibatches=16,
                 smoothing_coeff=0.005, eval_interval=1000, max_buffer_seq_len=100, optimizer=Adam):
        self.env = env
        self.device = device

        self.act_dim = env.act_dim
        self.input_dim = env.input_dim
        self.output_dim = env.output_dim
        self.obs_dim = self.input_dim + 2*self.output_dim

        self.train_start = int(train_start)
        self.n_train_steps = int(n_train_steps)

        data_net_hidden = 8
        self.data_net = TrainDataNet(self.input_dim, self.output_dim, data_net_hidden, self.device)

        self.controller = controller
        self.controller_latent = self.controller.latent_module
        self.controller_actor = self.controller.actor_module

        critic_hidden = int(2*controller.hidden_dim)
        self.critic1 = Critic(controller.hidden_dim, self.act_dim, critic_hidden, self.device)
        self.critic2 = Critic(controller.hidden_dim, self.act_dim, critic_hidden, self.device)
        self.value = Value(controller.hidden_dim, controller.hidden_dim, self.device)
        self.value_target = Value(controller.hidden_dim, controller.hidden_dim, self.device)

        self.buffer_size = int(buffer_size)
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dim, controller.hidden_dim, self.act_dim)
        self.latent_burn_in = latent_burn_in
        self.max_buffer_seq_len = max_buffer_seq_len

        self.sample_thresh = sample_thresh

        self.ep_min = ep_min
        self.ep_max = ep_max
        self.ep_len = 0
        self.train_x = None
        self.train_y = None

        self.lr_value = lr_value
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.smoothing_coeff = smoothing_coeff

        # self.controller_optimizer = optimizer(chain(self.controller.parameters(), self.pi.parameters()), lr=self.lr_actor)
        self.controller_optimizer = optimizer(self.controller.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optimizer(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optimizer(self.critic2.parameters(), lr=self.lr_critic)
        self.value_optimizer = optimizer(self.value.parameters(), lr=self.lr_value)

        self.mse_loss = nn.MSELoss()
        self.ent_coeff = ent_coeff

        self.action_pen = action_penalty_coeff

        self.n_minibatches = n_minibatches
        self.eval_interval = eval_interval

    def train(self):
        n_grad_steps = self.n_train_steps - self.train_start
        train_losses = np.zeros((n_grad_steps, 4, self.max_buffer_seq_len))
        target_losses = np.zeros(n_grad_steps)

        env_step = 0
        for step in range(self.n_train_steps):
            print(step)
            if env_step == 0:
                # randomize target network and generate new training data
                # if np.random.rand() < 0.1:  # probability of 0.1 of resetting target network weights
                #     self.env.reset()
                self._generate_training_data()
                print("reset")

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM and will not be stored in replay buffer
                # inputs = torch.zeros((1, self.obs_dim + self.act_dim + 3), device=self.device)
                inputs = torch.zeros((1, self.act_dim + 3), device=self.device)
                hidden = None
                cell = None

            with torch.no_grad():
                action, action_p, hidden, cell = self.controller.forward(inputs, hidden, cell)

                # action = torch.ones(self.act_dim)
                # action[[i for i in range(10) if i%2 == 1]] = 0
                # action = action.reshape((1, -1))

                action_ent = -torch.sum(action*torch.log(action_p)) - torch.sum((1-action)*torch.log(1-action_p))
                action_ent = action_ent.item()
                action_ent = np.array(action_ent, ndmin=2)

            # get "training data" for target network
            x = self.train_x[env_step].reshape((1, -1))
            y_target = self.train_y[env_step].reshape((1, -1))
            x_tensor = torch.from_numpy(x).float().to(self.device)
            y_targ_tensor = torch.from_numpy(y_target).float().to(self.device)
            y, loss = self.env.step((x_tensor, y_targ_tensor, action))
            y = y.detach().to('cpu').numpy()
            loss = loss.detach().to('cpu').numpy()

            obs = np.concatenate((x, y, y_target), axis=1)

            # change environment and target func controller must optimize
            env_step += 1
            done = (env_step == self.ep_len)
            if done:
                env_step = 0

            action = action.detach().to('cpu').numpy()
            if cell is not None:
                self.buffer.add(obs, hidden.detach().to('cpu').numpy(), cell.detach().to('cpu').numpy(), action, loss, done)
            else:
                self.buffer.add(obs, hidden.detach().to('cpu').numpy(), None, action, loss,
                                done)

            action_size = np.sum(action, axis=1).reshape((1, -1))

            loss = loss.reshape((1, -1))
            # inputs = np.concatenate((obs, action, loss, action_size, action_ent), axis=1)
            inputs = np.concatenate((action, loss, action_size, action_ent), axis=1)
            inputs = torch.from_numpy(inputs).float().to(self.device)

            if step >= self.train_start:
                value_losses, Q1_losses, Q2_losses, pi_losses = self._grad_update()

                seq_len = value_losses.size
                idx = step - self.train_start
                train_losses[step, 0, :seq_len] = value_losses
                train_losses[step, 1, :seq_len] = Q1_losses
                train_losses[step, 2, :seq_len] = Q2_losses
                train_losses[step, 3, :seq_len] = pi_losses

                target_losses[idx] = loss

                print(loss[0,0], value_losses[0], Q1_losses[0], Q2_losses[0], pi_losses[0])
                print(action_size[0,0])
                print(inputs)

                if step % self.eval_interval == 0:
                    self._evaluate()

        return train_losses, target_losses

    def _grad_update(self):
        # Compute gradients
        for batch in range(self.n_minibatches):
            obs_array, hidden, cell, action_array, loss_array, done_array = self.buffer.sample(self.max_buffer_seq_len)
            # hidden = torch.from_numpy(hidden.reshape(1, -1)).float().to(self.device)
            # cell = torch.from_numpy(cell.reshape(1, -1)).float().to(self.device)
            hidden = torch.zeros((1, self.controller.hidden_dim), device=self.device)
            cell = torch.zeros((1, self.controller.hidden_dim), device=self.device)

            sample_len = loss_array.size
            if sample_len > self.latent_burn_in:  # only perform burn-in if buffer returned sample length that's long enough
                burn_in_steps = self.latent_burn_in
            else:
                burn_in_steps = 0

            n_seq_steps = sample_len - burn_in_steps
            value_losses = np.zeros(n_seq_steps)
            Q1_losses = np.zeros(n_seq_steps)
            Q2_losses = np.zeros(n_seq_steps)
            pi_losses = np.zeros(n_seq_steps)

            # burn in of LSTM hidden states
            with torch.no_grad():
                for ii in range(burn_in_steps):
                    obs = torch.from_numpy(obs_array[ii].reshape((1, -1))).float().to(self.device)
                    action = torch.from_numpy(action_array[ii].reshape((1, -1))).float().to(self.device)
                    loss = torch.from_numpy(loss_array[ii].reshape((1, -1))).float().to(self.device)

                    # Note: action probabilities will change over course of training. To avoid staleness, we recompute the entropy term.
                    action_size = torch.sum(action, dim=1, keepdim=True)
                    _, action_p = self.controller_actor.forward(hidden)
                    action_ent = -torch.sum(action*torch.log(action_p)) - torch.sum((1-action)*torch.log(1-action_p))
                    action_ent = action_ent.reshape((1, -1))

                    # inputs = torch.concat((obs, action, loss, action_size, action_ent), dim=1)
                    inputs = torch.concat((action, loss, action_size, action_ent), dim=1)
                    hidden, cell = self.controller_latent.forward(inputs, hidden, cell)

                    if done_array[ii]:  # if starting new task, reinitialize latents to 0 and perform initial LSTM dummy step
                        # inputs = torch.zeros((1, self.obs_dim + self.act_dim + 3), device=self.device)
                        inputs = torch.zeros((1, self.act_dim + 3), device=self.device)
                        hidden = None
                        cell = None
                        # note: the input vec only used to sample actions, a functionality we don't care about here
                        hidden, cell = self.controller_latent.forward(inputs, hidden, cell)

            # gradient-update loop
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            self.controller_optimizer.zero_grad()

            cumulative_pi_loss = torch.tensor(0.)
            cumulative_value_loss = 0
            cumulative_Q1_loss = 0
            cumulative_Q2_loss = 0
            loss_idx = 0
            for ii in range(burn_in_steps, sample_len):
                obs = torch.from_numpy(obs_array[ii].reshape((1, -1))).float().to(self.device)
                action = torch.from_numpy(action_array[ii].reshape((1, -1))).float().to(self.device)
                loss = torch.from_numpy(loss_array[ii].reshape((1, -1))).float().to(self.device)

                # detach hidden state tensor for critic networks so that grads don't propagate through LSTM
                critic_hidden = hidden.detach()

                # compute target for V
                with torch.no_grad():
                    action_sample, action_p = self.controller_actor(critic_hidden)
                    action_ent = -torch.sum(action_sample*torch.log(action_p)) - torch.sum((1-action_sample)*torch.log(1-action_p))

                    Q1 = self.critic1(critic_hidden, action_sample)
                    Q2 = self.critic2(critic_hidden, action_sample)
                    Q = torch.minimum(Q1, Q2)
                    target_value = Q + self.ent_coeff * action_ent

                # track losses of V across trajectory (note: we accumulate losses then backprop at end, like a minibatch)
                value = self.value(critic_hidden)
                value_loss = self.mse_loss(value, target_value)
                value_losses[loss_idx] = value_loss.item()
                value_loss.backward()
                cumulative_value_loss += value_loss.item()
                # print('V: ', target_value.item(), value.item(), value_loss.item())

                # compute target for Q
                with torch.no_grad():
                    action_size = torch.sum(action, dim=1, keepdim=True)
                    Q_target = -loss - self.action_pen * action_size * action_size + self.value_target(critic_hidden)

                # print("Q target: ", -loss.item(), (-self.action_pen * action_size * action_size).item(), self.value_target(hidden).item())

                # track losses of Q across trajectory (note: we accumulate losses then backprop at end, like a minibatch)
                Q = self.critic1(critic_hidden, action)
                Q1_loss = self.mse_loss(Q, Q_target)
                Q1_losses[loss_idx] = Q1_loss.item()
                # print('Q1: ', Q_target.item(), Q.item(), Q1_loss.item())
                Q1_loss.backward()
                cumulative_Q1_loss += Q1_loss.item()

                Q = self.critic2(critic_hidden, action)
                Q2_loss = self.mse_loss(Q, Q_target)
                Q2_losses[loss_idx] = Q2_loss.item()
                # print('Q2: ', Q_target.item(), Q.item(), Q2_loss.item())
                Q2_loss.backward()
                cumulative_Q2_loss += Q2_loss.item()

                # update policy via reparametrization trick (like in SAC paper)
                # policy network : K unit rand vars + state --> coords in K-dim unit cube
                # backprop via straight-through grads
                # not Gumbel bc this treats every dim independent; otherwise need 2^K dim output for all possible combos

                action_sample, action_p = self.controller_actor(hidden)

                self.critic1.requires_grad_(False)
                self.critic2.requires_grad_(False)
                Q1 = self.critic1(critic_hidden, action_sample)
                Q2 = self.critic2(critic_hidden, action_sample)
                Q = torch.minimum(Q1, Q2)

                # LSTM weights to be updated only via action_sample
                neg_action_ent = torch.sum(action_sample * torch.log(action_p)) + torch.sum((1. - action_sample) * torch.log(1. - action_p))
                pi_loss = neg_action_ent - Q
                pi_losses[loss_idx] = pi_loss.item()
                cumulative_pi_loss = cumulative_pi_loss + pi_loss
                # print('pi_loss: ', pi_loss.item())
                # print(neg_action_ent.item(), Q.item(), action_sample.detach().numpy())
                # print('gate probs: ', (action_sample * action_p + (1. - action_sample) * (1. - action_p)).detach().numpy())
                # print('probs: ', action_p.detach().numpy())

                self.critic1.requires_grad_(True)
                self.critic2.requires_grad_(True)

                # Note: action probabilities will change over course of training. To avoid staleness, we recompute the entropy term.
                _, action_p = self.controller_actor(critic_hidden)
                action_ent = -torch.sum(action * torch.log(action_p)) - torch.sum((1. - action) * torch.log(1. - action_p))
                action_ent = action_ent.reshape((1, -1))

                # inputs = torch.concat((obs, action, loss, action_size, action_ent), dim=1)
                inputs = torch.concat((action, loss, action_size, action_ent), dim=1)

                if done_array[ii]:  # if starting new task, reinitialize latents to 0 and perform initial LSTM dummy step
                    # inputs = torch.zeros((1, self.obs_dim + self.act_dim + 3), device=self.device)
                    inputs = torch.zeros((1, self.act_dim + 3), device=self.device)
                    hidden = None
                    cell = None

                hidden, cell = self.controller_latent.forward(inputs, hidden, cell)

            print('pi loss: ', cumulative_pi_loss.item())
            print('Q1 loss: ', cumulative_Q1_loss)
            print('Q2 loss: ', cumulative_Q2_loss)
            print('value loss: ', cumulative_value_loss)
            cumulative_pi_loss.backward()

        # update weights
        self.value_optimizer.step()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        self.controller_optimizer.step()

        # update critic targets via moving average
        PolyakUpdate(self.value.parameters(), self.value_target.parameters(), self.smoothing_coeff)

        return value_losses, Q1_losses, Q2_losses, pi_losses

    def _evaluate(self):
        # used for evaluating training progress
        pass

    # def _generate_training_data(self):
    #     self.ep_len = np.random.randint(self.ep_min, self.ep_max + 1)
    #     self.train_x = np.zeros((self.ep_len, self.input_dim), dtype=np.float32)
    #     self.train_y = np.zeros((self.ep_len, self.output_dim), dtype=np.float32)
    #
    #     # set noise level for output
    #     max_std = 3
    #     y_noise_std = max_std * np.random.uniform()  # uniformly sample Gaussian noise variance
    #
    #     # randomly reset data-generating network weights
    #     std = np.random.uniform(1, 5)
    #     self.data_net.reset_weights(std)
    #
    #     x_std = 10
    #     with torch.inference_mode():
    #         for ii in range(self.ep_len):
    #             x_sample = x_std*torch.randn(size=(1, self.input_dim))
    #             y_sample = self.data_net(x_sample)
    #
    #             # add noise to train_y
    #             y_noise = y_noise_std * torch.randn(size=(1, self.output_dim))
    #             y_sample += y_noise
    #
    #             # store dataset
    #             self.train_x[ii] = x_sample.to('cpu').numpy()
    #             self.train_y[ii] = y_sample.to('cpu').numpy()

    def _generate_training_data(self):
        self.ep_len = np.random.randint(self.ep_len_min, self.ep_len_max + 1)
        self.train_x = np.zeros((self.ep_len, self.input_dim), dtype=np.float32)
        self.train_y = np.zeros((self.ep_len, self.output_dim), dtype=np.float32)

        # set noise level for output
        # max_std = 3
        max_std = 0
        y_noise_std = max_std * np.random.uniform()  # uniformly sample Gaussian noise variance

        # n_gates = np.random.randint(1, self.act_dim)
        # gate_idx = np.random.permutation(self.act_dim)
        # gate_idx = gate_idx[:n_gates]
        # gating = torch.zeros(self.act_dim, device=self.device)
        # gating[gate_idx] = 1

        gating = torch.ones(self.act_dim, device=self.device)
        # gating[[0,2,4,6,8]] = 0
        gating[1] = 0
        # print('n gates: ', n_gates)

        x_std = 10
        with torch.inference_mode():
            train_x = x_std * torch.randn(size=(self.ep_len, self.input_dim), device=self.device)
            train_y = self.env.gen_training_data(train_x, gating)

            # add noise to train_y
            y_noise = y_noise_std * torch.randn(size=(self.ep_len, self.output_dim), device=self.device)
            train_y.add_(y_noise)

            # store dataset
            self.train_x = train_x.to('cpu').numpy()
            self.train_y = train_y.to('cpu').numpy()
