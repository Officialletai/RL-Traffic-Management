#Reinforcement Learning, Multiagent, IQL DQN
import random
from environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
import os
import optuna

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# There will be as many agent as there are nodes
class NodeAgent:
    def __init__(self, input_size, output_size, node_number, discount_factor, learning_rate):

        # Hyperparameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
        self.environment = Environment()
        self.model = DQN(input_size,output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.node_number = node_number

    def pick_action(self, state, epsilon):
        if random.random() < epsilon:
            
            action = self.environment.pick_random_phase(self.node_number)
            return action
            
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            action = torch.argmax(q_values).item() + 1
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

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        #print("saved model")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

class MultiAgent:
    def __init__(self, environment, epsilon_decay_rate, discount_factor, learning_rate):
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.environment = environment
        self.num_nodes = self.environment.map.num_nodes
        self.agents = {}

        for node in range(self.num_nodes):
            degree = self.environment.map.nodes[str(node)].degree
            if degree == 1:
                degree = 2
            input_dimension = (2*(degree*degree))+degree #queue, traffic light, edge
            output_dimension = len(self.environment.controller.get_phase(degree))
            self.agents[node] = NodeAgent(input_dimension, output_dimension, node, self.discount_factor, self.learning_rate)

    def get_actions(self, local_states):
        actions = {}

        for node in self.agents:
            local_state = local_states[node]
            action = self.agents[node].pick_action(local_state, self.epsilon)
            actions[node] = action

        return actions 
        
    def update_agents(self, local_states, actions, rewards, next_local_states, finished, episode):
        for node in self.agents:
            local_state = local_states[node]
            action = actions[node]
            reward = rewards[node]
            next_local_state = next_local_states[node]

            self.agents[node].train(local_state, action, reward, next_local_state, finished)
        
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

    def save_models(self, save_dir):
        
        #print("almost about to save very close")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for node, agent in self.agents.items():
            agent.save_model(f"{save_dir}/agent_{node}.pth")

    def load_models(self, save_dir):

        if not os.path.exists(save_dir):
            raise ValueError(f"Directory {save_dir} does not exist.")
        
        for node, agent in self.agents.items():
            agent.load_model(f"{save_dir}/agent_{node}.pth")
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)


def train_multi_agent(episodes=1250, epsilon_decay_rate=0.995, discount_factor=0.995, learning_rate=0.0005):
    env = Environment()
    env.map.draw()
    multi_agent = MultiAgent(env, epsilon_decay_rate, discount_factor, learning_rate)
    times = []
    average_car_wait_time = []
    
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
        
        #multi_agent.decay_epsilon()
        times.append(env.time)

        avg_time = sum(env.time_in_traffic.values()) / len(env.time_in_traffic)
        average_car_wait_time.append(avg_time)

        if episode % 25 == 0 and episode != 0:
            print(
                f"Epochs: {episode}, average_car_wait_time: {np.mean(average_car_wait_time)}, lowest_car_wait_time: {np.amin(average_car_wait_time)}, epsilon_value: {multi_agent.epsilon}"
            ) 

            average_car_wait_time = []
            times = []
            multi_agent.save_models("dqn_models")
    
    return times

if __name__ == "__main__":

    def objective_function(trial):
        discount_factor = trial.suggest_float("discount_factor", 0.995, 0.9999, step=0.0001)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 0.999, 0.9999, step=0.0001)

        print(f"Training with discount_factor={discount_factor}, learning_rate={learning_rate}, epsilon_decay_rate={epsilon_decay_rate}")
        performance = train_multi_agent(2500, epsilon_decay_rate, discount_factor, learning_rate)
        
        return performance

    test = optuna.create_study(direction="maximize")
    test.optimize(objective_function, n_trials=100)

    best_params = test.best_params
    best_reward = test.best_value
    print(f"Best parameters: {best_params} with reward: {best_reward}")
 
