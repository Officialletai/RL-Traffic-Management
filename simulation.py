from environment import Environment
import numpy as np

simulation = Environment()

def print_current_state(simulation):

    normalised_weight_matrix, queue_matrix, edge_matrix, traffic_light_matrix = simulation.get_state()

    print('normalised weight matrix \n', normalised_weight_matrix)
    print('queue matrix \n', queue_matrix)
    print('edge matrix \n', edge_matrix)
    print('traffic light matrix \n', traffic_light_matrix)

print_current_state(simulation)

# each action is a node phase pairing
# the actions is a list of actions
action_1 = [(4,2)]


simulation.step(action_1)
simulation.step(action_1)
simulation.step(action_1)

