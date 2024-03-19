import numpy as np
from itertools import chain
from .model import TrainDataNet

import torch
import torch.nn as nn
from torch.optim import Adam


class PPO:
    '''
    Adapted from https://github.com/vwxyzjn/ppo-implementation-details,
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#Schulman2017,
    and https://github.com/lucaslingle/pytorch_rl2.
    '''

    def __init__(self, env, eval_env, controller, controller_input_size, device, max_grad_steps=2E5, ep_len_min=1E4, ep_len_max=1E6,
                 action_penalty_coeff=None, lbda=0.9, gamma=0.99, lr_critic=1E-1, lr_actor=1E-4,
                 n_meta_eps = 5E3, n_eps_per_task = 50, meta_ep_batch = 50, rollout_len=100,
                 epsilon=0.2, eval_interval=1000, n_ppo_epochs=8, optimizer=Adam):
        self.env = env
        self.eval_env = eval_env
        self.device = device

        self.act_dim = env.act_dim
        self.input_dim = env.input_dim
        self.output_dim = env.output_dim
        self.obs_dim = self.input_dim + 2*self.output_dim

        self.controller_input_size = controller_input_size

        data_net_hidden = 8
        # self.data_net = TrainDataNet(self.input_dim, self.output_dim, data_net_hidden, self.device)

        self.controller = controller

        self.max_grad_steps = int(max_grad_steps)
        self.n_ppo_epochs = int(n_ppo_epochs)
        self.epsilon = epsilon

        self.rollout_len = int(rollout_len)
        self.n_meta_eps = int(n_meta_eps)
        self.meta_ep_batch = int(meta_ep_batch)
        self.n_eps_per_task = int(n_eps_per_task)

        self.ep_len_min = ep_len_min
        self.ep_len_max = ep_len_max
        self.ep_len = None  # this will be set when training data for target network generated
        self.train_x = None
        self.train_y = None

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.lbda = lbda
        self.gamma = gamma

        self.optimizer = optimizer([{'params' : chain(self.controller.latent_module.parameters(),
                                                    self.controller.actor_module.parameters()),
                                     'lr' : self.lr_actor},
                                    {'params': self.controller.critic_module.parameters(),
                                     'lr': self.lr_critic}])

        self.mse_loss = nn.MSELoss()

        # default action penalty set as inverse square of action dim
        if action_penalty_coeff is None:
            self.action_pen = 1./(self.act_dim*self.act_dim)
        else:
            self.action_pen = action_penalty_coeff
        self.eval_interval = eval_interval

    def train(self):
        rollout_actions = np.zeros((self.n_meta_eps, self.rollout_len, self.act_dim), dtype=np.float)
        rollout_inputs = np.zeros((self.n_meta_eps, self.rollout_len, self.controller_input_size), dtype=np.float)
        rollout_rew = np.zeros((1, self.rollout_len), dtype=np.float)
        rollout_values = np.zeros((self.n_meta_eps, self.rollout_len+1), dtype=np.float)
        rollout_log_probs = np.zeros((self.n_meta_eps, self.rollout_len), dtype=np.float)
        rollout_advtg = np.zeros((self.n_meta_eps, self.rollout_len), dtype=np.float)
        rollout_returns = np.zeros((self.n_meta_eps, self.rollout_len), dtype=np.float)

        if self.controller.latent_type is nn.LSTM:
            initial_rnn_states = np.zeros((self.n_meta_eps, 2, self.controller.hidden_dim), dtype=np.float)
        else:
            initial_rnn_states = np.zeros((self.n_meta_eps, self.controller.hidden_dim), dtype=np.float)

        step = 0
        policy_losses = []
        value_losses = []
        target_losses = []

        while step < self.max_grad_steps:
            # PPO rollout phase

            for task_idx in range(0, self.n_meta_eps, self.n_eps_per_task):
                if np.random.rand() < 1:  # probability of 0.1 of resetting target network weights
                    self.env.reset()
                # randomize target network and generate new training data
                self._generate_training_data()

                # sample an initial, random action
                # note: first step uses dummy inputs to LSTM
                inputs = torch.zeros((1, self.controller_input_size), device=self.device)

                if self.controller.latent_type is nn.LSTM:
                    rnn_state = (torch.zeros((1, self.controller.hidden_dim), device=self.device),
                                 torch.zeros((1, self.controller.hidden_dim), device=self.device))
                else:
                    rnn_state = torch.zeros((1, self.controller.hidden_dim), device=self.device)

                for ep_idx in range(self.n_eps_per_task):
                    # store initial RNN latent states
                    if self.controller.latent_type is nn.LSTM:
                        initial_rnn_states[task_idx+ep_idx, 0] = rnn_state[0].to('cpu')
                        initial_rnn_states[task_idx+ep_idx, 1] = rnn_state[1].to('cpu')
                    else:
                        initial_rnn_states[task_idx+ep_idx] = rnn_state.to('cpu')

                    data_idx = np.random.permutation(self.ep_len)
                    self.train_x = self.train_x[data_idx]
                    self.train_y = self.train_y[data_idx]
                    for jj in range(self.rollout_len):
                        with torch.no_grad():
                            # print("rnn: ", rnn_state)
                            action, log_action_p, value, rnn_state = self.controller(inputs, rnn_state=rnn_state)

                        rollout_inputs[task_idx+ep_idx, jj] = inputs.detach().to('cpu').numpy()
                        rollout_actions[task_idx+ep_idx, jj] = action.detach().to('cpu').numpy()
                        rollout_log_probs[task_idx+ep_idx, jj] = log_action_p.to('cpu').numpy()
                        rollout_values[task_idx+ep_idx, jj] = value.to('cpu').numpy()

                        # get next (x,y) input and target for target network
                        x = self.train_x[jj].reshape((1, -1))
                        y_target = self.train_y[jj].reshape((1, -1))
                        x_tensor = torch.from_numpy(x).float().to(self.device)
                        y_targ_tensor = torch.from_numpy(y_target).float().to(self.device)
                        y, loss = self.env.step((x_tensor, y_targ_tensor, action))
                        # y = y.detach().to('cpu').numpy()
                        target_losses.append(loss.item())

                        # construct next input tensor
                        action_size = action.sum(dim=1, keepdim=True)
                        rew = -loss - self.action_pen * action_size * action_size
                        # loss = loss.reshape((1, -1))
                        obs = (y-y_targ_tensor).pow(2)
                        # inputs = torch.cat((obs, action, loss, action_size*action_size, rew), dim=1)
                        inputs = torch.cat((obs, action, action_size * action_size, rew), dim=1)

                        rollout_rew[0, jj] = rew.item()

                    rnn_state = rnn_state.detach()
                    _, _, next_value, _ = self.controller(inputs, rnn_state=rnn_state)
                    rollout_values[task_idx+ep_idx, -1] = next_value.detach().to('cpu').numpy()
                    advantages, returns = self._compute_advantages(rollout_rew, rollout_values[task_idx+ep_idx])
                    rollout_advtg[task_idx+ep_idx], rollout_returns[task_idx+ep_idx] = advantages, returns

            # training phase
            for _ in range(self.n_ppo_epochs):
                eps_idx = np.random.permutation(self.n_meta_eps)
                for ii in range(0, self.n_meta_eps, self.meta_ep_batch):
                    mb_eps_idx = eps_idx[ii:ii+self.meta_ep_batch]

                    mb_initial_rnn = torch.from_numpy(initial_rnn_states[mb_eps_idx]).float().to(self.device)
                    mb_rollout_actions = torch.from_numpy(rollout_actions[mb_eps_idx]).float().to(self.device)
                    mb_rollout_inputs = torch.from_numpy(rollout_inputs[mb_eps_idx]).float().to(self.device)
                    mb_rollout_old_log_probs = torch.from_numpy(rollout_log_probs[mb_eps_idx]).float().to(self.device)
                    mb_rollout_advtg = torch.from_numpy(rollout_advtg[mb_eps_idx]).float().to(self.device)
                    mb_rollout_returns = torch.from_numpy(rollout_returns[mb_eps_idx]).float().to(self.device)

                    _, new_log_probs, new_values, _ = self.controller(mb_rollout_inputs, rnn_state=mb_initial_rnn, action=mb_rollout_actions)

                    # compute ratio and clipped ratio
                    ratio = (new_log_probs - mb_rollout_old_log_probs).exp()
                    clipped_ratio = torch.clip(ratio, 1.-self.epsilon, 1.+self.epsilon)

                    self.optimizer.zero_grad()

                    # policy loss
                    policy_loss = -torch.minimum(ratio*mb_rollout_advtg, clipped_ratio*mb_rollout_advtg).mean()
                    policy_losses.append(policy_loss.item())
                    policy_loss.backward()

                    # value loss (optimized separately from policy loss)
                    #  - only policy optimization shapes RNN computations; we do not let gradients propagate from values back into RNN
                    value_loss = self.mse_loss(new_values, mb_rollout_returns)
                    value_losses.append(value_loss.item())
                    value_loss.backward()

                    self.optimizer.step()
                    step += 1

            print("losses: ", policy_losses[-1], value_losses[-1], np.mean(target_losses[-self.rollout_len * self.n_meta_eps:]), rollout_rew.mean())
            # print("losses: ", policy_losses[-1], value_losses[-1], np.mean(target_losses[-self.rollout_len*self.n_meta_eps:]), rollout_rew.mean(), np.exp(rollout_log_probs[-1,-1]))
            # print("final action: ", rollout_actions[:, -1, :])

            if step % self.eval_interval == 0:
                for _ in range(5):
                    self._evaluate()

        return policy_losses, value_loss, target_losses

    def _compute_advantages(self, rollout_rew, rollout_values):
        TD_errs = rollout_rew + self.gamma*rollout_values[1:] - rollout_values[:-1]

        advantages = np.zeros_like(TD_errs)
        advantages[-1] = TD_errs[-1]
        for ii in reversed(range(len(advantages)-1)):
            advantages[ii] = TD_errs[ii] + self.gamma*self.lbda*advantages[ii+1]

        returns = advantages + rollout_values[:-1]
        return advantages, returns

    def _evaluate(self):
        # used for evaluating training progress
        self.eval_env.reset()
        eval_x, eval_y, gating = self._generate_data(ep_len=self.rollout_len, generating_fn=self.eval_env.gen_training_data)
        print("eval gating: ", gating)

        inputs = torch.zeros((1, self.controller_input_size), device=self.device)
        if self.controller.latent_type is nn.LSTM:
            rnn_state = (torch.zeros((1, self.controller.hidden_dim), device=self.device),
                         torch.zeros((1, self.controller.hidden_dim), device=self.device))
        else:
            rnn_state = torch.zeros((1, self.controller.hidden_dim), device=self.device)

        losses = []
        for ii in range(self.rollout_len):
            with torch.no_grad():
                # print("rnn: ", rnn_state)
                # print("input: ", inputs)
                action, log_action_p, value, rnn_state = self.controller(inputs, rnn_state=rnn_state)

            # get next (x,y) input and target for target network
            x = eval_x[ii].reshape((1, -1))
            y_target = eval_y[ii].reshape((1, -1))
            x_tensor = torch.from_numpy(x).float().to(self.device)
            y_targ_tensor = torch.from_numpy(y_target).float().to(self.device)
            y, loss = self.eval_env.step((x_tensor, y_targ_tensor, action))
            losses.append(loss.item())
            print("eval loss: ", ii, losses[-1])
            print("action: ", action.detach().to("cpu").numpy())

            # construct next input tensor
            action_size = action.sum(dim=1, keepdim=True)
            rew = -loss - self.action_pen * action_size * action_size
            obs = (y - y_targ_tensor).pow(2)
            inputs = torch.cat((obs, action, action_size * action_size, rew), dim=1)

            _, gating_log_prob, _, _ = self.controller(inputs, rnn_state=rnn_state, action=gating)
            print("gating prob: ", gating_log_prob.exp().item())

        print("mean eval loss: ", np.mean(losses))

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
        self.train_x, self.train_y, _ = self._generate_data(self.ep_len, self.env.gen_training_data)

    def _generate_data(self, ep_len, generating_fn):
        # set noise level for output
        max_std = 3
        y_noise_std = max_std * np.random.uniform()  # uniformly sample Gaussian noise variance

        n_gates = np.random.randint(1, self.act_dim)  # select rand number of gates to open
        gate_idx = np.random.permutation(self.act_dim)
        gate_idx = gate_idx[:n_gates]  # randomly select gate indices
        gating = torch.zeros(self.act_dim, device=self.device)
        gating[gate_idx] = 1  # set selected gates to be open

        # gating = torch.ones(self.act_dim, device=self.device)
        # gating[1] = 0

        x_std = 10
        with torch.inference_mode():
            train_x = x_std * torch.randn(size=(ep_len, self.input_dim), device=self.device)
            train_y = generating_fn(train_x, gating)

            # add noise to train_y
            y_noise = y_noise_std * torch.randn(size=(ep_len, self.output_dim), device=self.device)
            train_y.add_(y_noise)

            # store dataset
            data_x = train_x.to('cpu').numpy()
            data_y = train_y.to('cpu').numpy()

        return data_x, data_y, gating
