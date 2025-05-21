import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_layer_1_dim=64, hidden_layer_2_dim=32
    ):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_1_dim)
        self.fc2 = nn.Linear(hidden_layer_1_dim, hidden_layer_2_dim)
        self.out = nn.Linear(hidden_layer_2_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.out(x)


class RLAgent:
    def __init__(self, state_size=6, action_size=3):
        self.model = DQN(state_size, action_size)
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(3)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward + (
                self.gamma
                * torch.max(
                    self.model(torch.FloatTensor(next_state).unsqueeze(0))
                ).item()
                if not done
                else 0
            )
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            output = self.model(torch.FloatTensor(state).unsqueeze(0))
            loss = self.loss_fn(output, target_f.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
