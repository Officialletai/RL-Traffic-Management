#Reinforcement Learning, Multiagent, IQL DQN
import random
from environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools

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
    def __init__(self, input_size, output_size, node_number, epsilon):

        # Hyperparameters
        self.discount_factor = 0.995
        self.learning_rate = 0.001
        self.epsilon = epsilon
        
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(int(done), dtype=torch.float32).unsqueeze(0)

        # Compute Q(s, a)
        current_q_values = self.model(state)
        current_q_value = current_q_values.max(1)[0]

        # Compute Q(s', a')
        with torch.no_grad():
            next_q_values = self.model(next_state)
            next_q_value = next_q_values.max(1)[0]

        target = reward + (1 - done) * self.discount_factor * next_q_value

        loss = F.mse_loss(current_q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class MultiAgent:
    def __init__(self, environment):
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay_rate = 0.99975

        self.environment = environment
        self.num_nodes = self.environment.map.num_nodes
        self.agents = {}

        for node in range(self.num_nodes):
            degree = self.environment.map.nodes[str(node)].degree
            input_dimension = (2*(degree*degree))+degree #queue, traffic light, edge
            output_dimension = len(self.environment.controller.get_phase(degree))
            self.agents[node] = NodeAgent(input_dimension, output_dimension, node, self.epsilon)

    def get_actions(self, local_states):
        actions = {}

        for node in self.agents:
            local_state = local_states[node]
            action = self.agents[node].pick_action(local_state)
            actions[node] = action
        
        return actions 
        
    def update_agents(self, local_states, actions, rewards, next_local_states, finished, episode):
        for node in self.agents:
            local_state = local_states[node]
            action = actions[node]
            reward = rewards[node]
            next_local_state = next_local_states[node]
            if self.environment.map.intersections[self.agents[node].node_number] == 1:
                continue
            self.agents[node].train(local_state, action, reward, next_local_state, finished)
        
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)


def train_multi_agent(episodes=100):
    env = Environment()
    multi_agent = MultiAgent(env)
    times = []
    
    for episode in range(episodes):
        env.reset()
        
        local_states = env.get_local_state()

        finished = False

        while not finished:
            actions = multi_agent.get_actions(local_states)
            # current dictionary:
            # {1: 1, 2:3, 3:1, 4:6}
            # actions -> [(1,1), (2,4), (3,2)]
            actions_array = list(actions.items())
            next_local_states, rewards, finished = env.step(actions_array)

            multi_agent.update_agents(local_states, actions, rewards, next_local_states, finished, episode)
            local_states = next_local_states
        
        times.append(env.time)

        if episode % 25 == 0 and episode != 0:
            print(
                f"Epochs: {episode}, average_time: {np.mean(times)}, highest_score: {np.amin(times)}, epsilon_value: {multi_agent.epsilon}"
            ) 

            times = []
    
    return times

if __name__ == "__main__":

    # Define potential values for hyperparameters
    discount_factors = [0.9, 0.95, 0.99, 0.995]
    learning_rates = [0.001, 0.002, 0.005, 0.01]
    epsilon_decay_rates = [0.995, 0.9975, 0.999, 0.99975]

    # Create a list of all combinations of hyperparameters
    hyperparameters = list(itertools.product(discount_factors, learning_rates, epsilon_decay_rates))

    # Loop over each combination of hyperparameters and train your model
    for discount_factor, learning_rate, epsilon_decay_rate in hyperparameters:
        print(f"Training with discount_factor={discount_factor}, learning_rate={learning_rate}, epsilon_decay_rate={epsilon_decay_rate}")
    
        # Update the values in your code
        NodeAgent.discount_factor = discount_factor
        NodeAgent.learning_rate = learning_rate
        MultiAgent.epsilon_decay_rate = epsilon_decay_rate

        train_multi_agent()


        
