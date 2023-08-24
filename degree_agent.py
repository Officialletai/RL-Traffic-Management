#Reinforcement Learning, Multiagent, IQL DQN
from collections import deque
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
class DegreeAgent:
    def __init__(self, input_size, output_size, num_degree, discount_factor, learning_rate, environment, target_update_freq = 10):

        # Hyperparameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.train_steps = 0
        
        self.environment = environment
        self.model = DQN(input_size,output_size)
        self.target_model = DQN(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.num_degree = num_degree

    def pick_action(self, state, epsilon):
        if random.random() < epsilon:
            action = self.environment.pick_random_phase_degree(self.num_degree)
            return action
            
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            action = torch.argmax(q_values).item() + 1
            return action

    def train(self, memory):

        states, actions, rewards, next_states, dones = zip(*memory)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Compute Q(s, a)
        current_q_values = self.model(states).gather(1, (actions.unsqueeze(-1)-1)).squeeze(-1)

        # Compute Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.max(1)[0]

        targets = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = F.mse_loss(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        #print("saved model")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

class MultiAgent:
    def __init__(self, environment, epsilon_decay_rate, discount_factor, learning_rate, target_update_freq = 10):
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        self.environment = environment
        self.num_nodes = self.environment.map.num_nodes
        self.agents = {}

        for index in range(1, 5):
            degree = index
            if degree == 1:
                degree = 2
            input_dimension = (2*(degree*degree))+degree #queue, traffic light, edge
            output_dimension = len(self.environment.controller.get_phase(degree))
            self.agents[index] = DegreeAgent(input_dimension, output_dimension, degree, self.discount_factor, self.learning_rate, self.environment, self.target_update_freq)

        self.degrees = {}
        # EG: {4: [0, 1, 2, 3, 5, 6, 7], 2: [4], 3: [8], 1: [9]}

        for index in range(self.num_nodes):
            degree = self.environment.map.intersections[index]
            if degree in self.degrees.keys():
                self.degrees[degree].append(index)
            else:
                self.degrees[degree] = [index]


    def get_actions(self, local_states, environment):
        actions = {}

        for node in range(self.num_nodes):

            local_state = local_states[node]
            degree = environment.map.intersections[node]
            action = self.agents[degree].pick_action(local_state, self.epsilon)
            actions[node] = action

        return actions 
        
    def update_agents(self, local_states, actions, rewards, next_local_states, finished, episode):
        
        for agent in self.degrees:
            memory = []

            for node in self.degrees[agent]:
                local_state = local_states[node]
                action = actions[node]
                reward = rewards[node]
                next_local_state = next_local_states[node]
                memory.append((local_state, action, reward, next_local_state, finished))

            self.agents[agent].train(memory)
        
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


def train_multi_agent(num_nodes=10, sparsity_dist=[0.35, 0.65], num_cars=10, episodes=1250, epsilon_decay_rate=0.995, discount_factor=0.995, learning_rate=0.0005, trial=None, patience=3, target_update_freq = 10, validation_episodes=5):
    env = Environment(num_nodes, sparsity_dist, num_cars)
    env.map.draw()


    # for validation, we swap the sparsity distribution around for variance
    validation_env = Environment(num_nodes=num_nodes, sparsity_dist=sparsity_dist, num_cars=num_cars, seed=23082023)
    validation_env.map.draw()


    multi_agent = MultiAgent(env, epsilon_decay_rate, discount_factor, learning_rate, target_update_freq)
    times = []
    average_car_wait_time = []
    recent_avg_wait_times = deque(maxlen=5)

    validation_prior = [999]
    validation_post = []

    patience_counter = 0
    
    for episode in range(episodes):
        env.reset()
        local_states = env.get_local_state()

        finished = False

        while not finished:
            actions = multi_agent.get_actions(local_states, env)
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
        recent_avg_wait_times.append(np.mean(average_car_wait_time))
        smoothed_avg_wait_time = np.mean(recent_avg_wait_times)

        if episode % 25 == 0 and episode != 0:

            if trial:
                trial.report(smoothed_avg_wait_time, episode)
                if episode > 300:
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # Now to validate the model 
            last_validation = 0
            validation_average_car_wait_time = []
            num_validation_episodes = validation_episodes

            for _ in range(num_validation_episodes):
                validation_env.reset()
                local_states = validation_env.get_local_state()
                finished = False 

                while not finished:
                    actions = multi_agent.get_actions(local_states, validation_env)
                    actions_array = list(actions.items())
                    next_local_states, rewards, finished = validation_env.step(actions_array)
                    # Skip training step
                    local_states = next_local_states

                average_validation_wait_time = sum(validation_env.time_in_traffic.values()) / len(validation_env.time_in_traffic)
                validation_average_car_wait_time.append(average_validation_wait_time)
            last_validation = np.mean(validation_average_car_wait_time)
            validation_post.append(last_validation)

            print(
                f"Epochs: {episode}, average_car_wait_time: {np.mean(average_car_wait_time):.3f}, validation_wait_time: {last_validation:.3f}, epsilon_value: {multi_agent.epsilon:.6f}, "
            )

            patience_counter += 1
            
            if patience_counter >= patience:
                validation_avg_post = np.mean(validation_post)
                validation_avg_prior = np.mean(validation_prior)
                print(f'validation_avg_prior: {validation_avg_prior:.5f} validation_avg_post : {validation_avg_post:.5f}')

                if validation_avg_post > validation_avg_prior:
                    print(f"Early stopping at episode {episode} due to no improvement in validation performance.")
                    break

                else:
                    validation_prior = validation_post
                    validation_post = []

                    patience_counter = 0

            average_car_wait_time = []
            times = []
            multi_agent.save_models("dqn_models")
                                
    
    return smoothed_avg_wait_time

if __name__ == "__main__":
    NUM_NODES = 20
    SPARSITY_DIST=[73, 7]
    NUM_CARS=100
    EPISODES=2500
    N_TRIALS=10
    PATIENCE=4
    TARGET_UPDATE_FREQ=10
    VALIDATION_EPISODES=5

    def objective_function(trial):
        discount_factor = trial.suggest_float("discount_factor", 0.996, 0.99999, step=0.00075)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 0.9995, 0.99999, step=0.000075)

        print(f"Trial={trial.number}, Training with discount_factor={discount_factor}, learning_rate={learning_rate}, epsilon_decay_rate={epsilon_decay_rate}")
        performance = train_multi_agent(
            num_nodes=NUM_NODES, 
            sparsity_dist=SPARSITY_DIST, 
            num_cars=NUM_CARS, 
            episodes=EPISODES, 
            epsilon_decay_rate=epsilon_decay_rate, 
            discount_factor=discount_factor, 
            learning_rate=learning_rate,
            patience=PATIENCE,
            target_update_freq=TARGET_UPDATE_FREQ,
            validation_episodes=VALIDATION_EPISODES
            )
        
        return performance

    test = optuna.create_study(direction="minimize")
    test.optimize(objective_function, n_trials=N_TRIALS)

    best_params = test.best_params
    best_reward = test.best_value
    print(f"Best parameters: {best_params} with reward: {best_reward}")