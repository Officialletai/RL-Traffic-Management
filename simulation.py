from environment import Environment
import numpy as np

simulation = Environment()

def print_current_state(simulation):

    normalised_weight_matrix, queue_matrix, edge_matrix, traffic_light_matrix = simulation.get_state()

    print('normalised weight matrix \n', normalised_weight_matrix)
    print('queue matrix \n', queue_matrix)
    print('edge matrix \n', edge_matrix)
    print('traffic light matrix \n', traffic_light_matrix)

def random_actions(simulation):
    actions = []

    for i in simulation.controller.node_move_set:
        intersections = simulation.map.intersections[i]
        possible_moves = len(simulation.controller.get_phase(intersections))
        #print(possible_moves)
        random_move = np.random.choice(range(1, possible_moves + 1))
        actions.append((i, random_move))
    
    print(actions)
    return actions

for node, action in random_actions(simulation):
    simulation.controller.change_traffic_lights(node, action)

print_current_state(simulation)

finished = False

while not finished:
    random_action = random_actions(simulation)
    state, score, finished, time = simulation.step(random_action)

print(state, score, finished, time)
