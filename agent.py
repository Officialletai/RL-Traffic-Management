#Reinforcement Learning, Multiagent, IQL DQN
import random
from environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# There will be as many agent as there are nodes
class NodeAgent:
    def __init__(self, input_size, output_size, node_number):

        # Hyperparameters
        self.discount_factor = 0.995
        self.learning_rate = 0.05
        
        self.environment = Environment()
        self.model = DQN(input_size,output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.node_number = node_number

    def pick_action(self, state):
        if random.random() < self.epsilon:
            return self.environment.pick_random_phase(self.node_number)
        else:
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            return action

    def train(self, state, action, reward, next_state, done):
        
        # Compute Q(s, a)
        current_q_value = self.model(state)

        # Compute Q(s', a')
        with torch.no_grad():
            next_q_value = self.model(next_state)

        target = reward + (1 - done) * self.discount_factor * next_q_value

        loss = F.mse_loss(current_q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class MultiAgent:
    def __init__(self, local_states, rewards):
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay_rate = 0.999
        self.max_epochs = 10000

        self.agents = {}

    
    def get_actions(self):
        actions = []

    
    def update_agents(self):
        pass



def train_multi_agent(episodes=1000):
    env = Environment()
    multi_agent = MultiAgent()

    for episode in range(episodes):
        env.reset()
        
        local_states = env.get_local_state()

        finished = False

        while not finished:
            actions = multi_agent.get_actions()
            next_local_states, rewards, finished = env.step(actions)

            multi_agent.update_agents(local_states, actions, rewards, next_local_states, finished, episode)
            local_states = next_local_states
    


        
