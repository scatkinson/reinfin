import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging


class ReplayBuffer:

    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.state_memory = np.zeros(
            (self.mem_size, np.prod(input_shape)), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.mem_size, np.prod(input_shape)), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingDeepQNetwork(nn.Module):

    def __init__(
        self,
        lr,
        n_actions,
        name,
        input_dims,
        chkpt_dir,
        hid_out_dims=[512],
        dropout_size_list=[0.5],
    ):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, f"{name}.pt")

        self.hidden_layers_list = []
        self.dropout_list = []
        for i, hid_out_dim in enumerate(hid_out_dims):
            if i == 0:
                self.hidden_layers_list.append(
                    nn.Linear(in_features=input_dims, out_features=hid_out_dim)
                )
            else:
                self.hidden_layers_list.append(
                    nn.Linear(in_features=hid_out_dims[i - 1], out_features=hid_out_dim)
                )

        for dropout_size in dropout_size_list:
            self.dropout_list.append(nn.Dropout(dropout_size))
        # value stream
        self.V = nn.Linear(in_features=hid_out_dims[-1], out_features=1)
        # advantage function
        self.A = nn.Linear(in_features=hid_out_dims[-1], out_features=n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        flat1 = state
        for fc, dropout in zip(self.hidden_layers_list, self.dropout_list):
            flat1 = F.relu(fc(flat1))
            flat1 = dropout(flat1)
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        logging.info(f"Saving Checkpoint at {self.checkpoint_file}")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info(
            "Loading Checkpoint (must provide previous training run's pipeline_id in config)"
        )
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        hid_out_dims,
        dropout_size_list,
        batch_size,
        mem_size=1000000,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        chkpt_dir="/tmp",
        pipeline_id="",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.n_features = np.prod(input_dims)
        self.hid_out_dims = hid_out_dims
        self.dropout_size_list = dropout_size_list
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.n_features,
            name=f"q_eval_{pipeline_id}",
            chkpt_dir=self.chkpt_dir,
            hid_out_dims=self.hid_out_dims,
            dropout_size_list=self.dropout_size_list,
        )

        self.q_next = DuelingDeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.n_features,
            name=f"q_next_{pipeline_id}",
            chkpt_dir=self.chkpt_dir,
            hid_out_dims=self.hid_out_dims,
            dropout_size_list=self.dropout_size_list,
        )

    def choose_action(self, observation):
        if np.random.random() >= self.epsilon:
            state = T.tensor(
                np.array([observation], dtype=np.float32), dtype=T.float32
            ).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones.to(bool)] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()

        self.q_eval.optimizer.step()

        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss.item()
