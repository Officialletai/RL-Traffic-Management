#Reinforcement Learning, Multiagent, IQL DQN
from environment import Environment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class NodeAgent:
    def __init__(self, input_size, output_size):

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay_rate = 0.999
        self.max_epochs = 1000
        self.discount_factor = 0.995
        self.learning_rate = 0.05
        
        self.environment = Environment
        self.model = DQN(input_size,output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def pick_action(self, state):
        pass

    def train(self, state, action, reward, next_state, done, steps):
        pass

    def dqn_learn(self):
        pass