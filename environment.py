from controller import Controller
from map import Map
from car import Car
from node import Node

import numpy as np
import random


class Environment:
    def __init__(self, num_nodes=10, sparsity_dist=[0.35, 0.65], num_cars=10, seed=28062023):
        """
        Initializes the environment.
        
        Attributes:
        - map (Map): An instance of the Map class representing the environment layout.
        - cars (list): List of Car objects within the environment.
        - controller (Controller): Controls the traffic light system.
        - time (int): Keeps track of the simulation time.
        - score (int): Evaluation metric, quantifies how well cars navigate the environment.
        """
        self.num_nodes = num_nodes
        self.sparsity_dist = sparsity_dist
        self.seed = seed
        self.map = Map(num_nodes=self.num_nodes, sparsity_dist=self.sparsity_dist, seed=self.seed)
        
        self.num_cars = num_cars
        self.cars = self.initialise_cars(self.num_cars)
        self.controller = Controller(self.map)

        self.time = 0
        self.score = 0
        self.reward_array = {node_number:0 for node_number in range(self.map.num_nodes)}
        self.local_states = {}
        self.time_in_traffic = {car_number:0 for car_number in range(self.num_cars)}

        self.all_car_journeys = {str(car): [] for car in range(self.num_cars)}


    def initialise_cars(self, num_cars):
        """
        Initializes the cars in the environment.
        
        Args:
        - num_cars (int): Number of cars to be initialized.

        Returns:
        - list: List of Car objects.
        """
        cars_list = []
        for index in range(num_cars):
            start = random.randrange(0, self.map.num_nodes)
            stop = random.randrange(0, self.map.num_nodes)
            
            while stop == start:
                stop = random.randrange(0, self.map.num_nodes)
            
            car = Car(index, self.map, start, stop)

            # car.initialise_on_queue()

            cars_list.append(car)
            

        return cars_list


    def reset(self):
        """
        Resets the environment to its initial state.
        
        Returns:
        - tuple: Represents the state of the environment after reset.
        """
        self.map = Map(num_nodes=self.num_nodes, sparsity_dist=self.sparsity_dist, seed=self.seed)
        self.cars = self.initialise_cars(num_cars=self.num_cars)
        self.controller = Controller(self.map)

        self.time = 0
        self.score = 0
        self.reward_array = {node_number:0 for node_number in range(self.map.num_nodes)}
        self.time_in_traffic = {car_number:0 for car_number in range(self.num_cars)}


    def get_state(self):
        """
        Fetches the current state of the environment.
        
        Returns:
        - tuple: Tuple containing matrices representing the state of the environment.
        """
        # adjacency matrix weights
        weight_matrix = self.map.weight_matrix
        max_weight = weight_matrix.max()
        normalised_weight_matrix = weight_matrix / max_weight

        # car on nodes and edges matrix
        queue_matrix = np.zeros((self.map.num_nodes, self.map.num_nodes))
        edge_matrix = np.zeros((self.map.num_nodes, self.map.num_nodes))

        for car in self.cars:
            # if a car is at its destination, no longer include it in the state
            if car.finished == True:
                continue
            
            # if the car is on the edge, we can add it to the edge matrix
            # otherwise add it to the queue matrix
            if car.on_edge == True:
                edge_matrix[car.previous][car.next] += 1
            else:
                queue_matrix[car.current][car.next] += 1
        
        # traffic light matrix
        max_degree = max([self.map.nodes[str(node)].degree for node in self.map.nodes])
        traffic_light_matrix = np.zeros((self.map.num_nodes, max_degree*(max_degree-1)))
        
        for node in range(self.map.num_nodes):
            nodal_traffic_lights = self.map.nodes[str(node)].traffic_lights
            node_legal_journeys_counter = 0

            # if len(nodal_traffic_lights) == 1:
            #     node_legal_journeys_counter += 1
            #     continue

            for current_edge in nodal_traffic_lights:
                for next_edge in nodal_traffic_lights[current_edge]:
                    traffic_light_matrix[node, node_legal_journeys_counter] = nodal_traffic_lights[current_edge][next_edge].state
                    node_legal_journeys_counter += 1
        
        return normalised_weight_matrix, queue_matrix, edge_matrix, traffic_light_matrix
    

    def get_local_state(self):
        for agent_node in range(self.map.num_nodes):
                
            node = self.map.nodes[str(agent_node)]
            edge_labels = node.edge_labels

            # Initialise separate empty state spaces
            if node.degree == 1:
                local_queue_matrix = np.zeros((node.degree+1, node.degree+1))
                local_edge_vector = np.zeros(node.degree+1)
                local_traffic_lights = np.zeros((node.degree+1, node.degree+1))
            else:
                local_queue_matrix = np.zeros((node.degree, node.degree))
                local_edge_vector = np.zeros(node.degree)
                local_traffic_lights = np.zeros((node.degree, node.degree))

            for car in self.cars:
                # If a car is at its destination, no longer include it in the state
                if car.finished == True:
                    continue

                # If the car is on an edge and it is coming towards the node
                if car.on_edge == True and car.next == agent_node:
                    # In local terms, aka in A B C D, get the data of the car
                    # label_previous = node.edge_labels.get(str(car.previous))
                    label_previous = edge_labels.get(str(car.previous))

                    # If the label exists
                    if label_previous:
                        # Convert to A,B,C,D to 0,1,2,3 for matrix index
                        previous_pos = ord(label_previous) - ord('A')

                        # Add to appropriate position in edge vector
                        local_edge_vector[previous_pos] += 1

                # Else the car must be in a queue
                elif car.on_edge == False and car.current == agent_node:
                    # In local terms, aka in A B C D, get the data of the car
                    # label_previous = node.edge_labels.get(str(car.previous))
                    # label_next = node.edge_labels.get(str(car.next))
                    label_previous = edge_labels.get(str(car.previous))
                    label_next = edge_labels.get(str(car.next))

                    # If the label exists, add to edge vector
                    if label_previous and label_next:
                        # Convert to A,B,C,D to 0,1,2,3 for matrix index
                        previous_pos = ord(label_previous) - ord('A')
                        next_pos = ord(label_next) - ord('A')

                        # Add to appropriate position in queue matrix
                        local_queue_matrix[previous_pos, next_pos] += 1

            # Get the traffic light states for the node in 1s and 0s
            traffic_light_states = node.get_traffic_light_states()
                    
            for prev_node in edge_labels:
                for next_node in edge_labels:
                    prev_node_label = edge_labels[prev_node]
                    next_node_label = edge_labels[next_node]

                    # Translate previous and next node numbers/labels to indices for filling the matrix
                    prev_node_label_index = ord(prev_node_label) - ord('A')
                    next_node_label_index = ord(next_node_label) - ord('A')

                    # Diagonal is forced to 0 i.e. we don't allow backtracking or U-turns
                    if prev_node_label_index == next_node_label_index:
                        local_traffic_lights[prev_node_label_index][next_node_label_index] = 0
                    else:
                        local_traffic_lights[prev_node_label_index][next_node_label_index] = traffic_light_states[prev_node_label][next_node_label]

            # Normalise the state observation
            #n_local_queue_matrix = local_queue_matrix / np.linalg.norm(local_queue_matrix, axis=1, keepdims=True)
            #n_local_edge_vector = local_edge_vector / np.linalg.norm(local_edge_vector, axis=1, keepdims=True)

            # Normalise the local queue matrix
            local_queue_norm = np.linalg.norm(local_queue_matrix, axis=1, keepdims=True)
            mask = (local_queue_norm != 0)
            n_local_queue_matrix = np.where(mask, local_queue_matrix / local_queue_norm, local_queue_matrix)

            # Normalise the local edge vector
            local_edge_norm = np.linalg.norm(local_edge_vector, keepdims=True)
            mask = (local_edge_norm != 0)
            n_local_edge_vector = np.where(mask, local_edge_vector / local_edge_norm, local_edge_vector)
            
            # Create the state by combining and flattening all observations together into one array
            state= np.concatenate([n_local_queue_matrix.flatten(), n_local_edge_vector.flatten(), local_traffic_lights.flatten()])

            self.local_states[agent_node] = state

        return self.local_states
    

    def pick_random_phase(self, node):
        num_intersection = self.map.intersections[node]
        possible_phases = range(1, len(self.controller.get_phase(num_intersection)) + 1)
        random_phase = np.random.choice(possible_phases)
        return random_phase


    def pick_random_phase_degree(self, degree):
        possible_phases = range(1, len(self.controller.get_phase(degree)) + 1)
        random_phase = np.random.choice(possible_phases)
        return random_phase


    def step(self, actions):
        """
        Advances the simulation by one step. This function updates the state 
        of the environment based on the actions provided.
        
        Args:
        - actions (list of tuples): Each tuple contains a node and a phase number to set the traffic light to.
        
        Returns:
        - tuple: New state of the environment, current score, and boolean indicating if simulation is finished.
        """
        self.reward_array = {node_number:0 for node_number in range(self.map.num_nodes)}
        self.local_states = {}
        finished = True

        for node, phase in actions:
            self.controller.change_traffic_lights(node, phase)


        car_count = 0
        for car in self.cars[:]:
            if car.finished:
                #print(f'car {car.id} has finished')
                self.score += car.reward
                self.cars.remove(car)
                continue
            else:
                finished = False

            current_node = car.current
            previous_node = car.previous
            next_node = car.next
            # if car is not edge, it must be in queue
            # if car is on road and has more intersections to get through, check them
            # if no more intersections after the coming node, it is moving towards destination
            # it will just join any queue and end there.
            if car.on_edge:
                
                # As soon as the car enters its first edge, it has officially set off. It remains True for the rest of the journey
                car.set_off = True

                self.all_car_journeys[str(car.id)].append([(int(previous_node), int(next_node)), car.road_progress/100])

                current_node_labels = self.map.nodes[str(next_node)].edge_labels
                                
                previous_node_label = current_node_labels[str(previous_node)]

                # if there are more nodes to go through, check the next next node
                # otherwise, this is the final destination, so move into the node and finish
                if len(car.path) > 1:
                    next_node = car.path[1]
                else:
                    car.move(True)
                    continue

                #print('next node:', next_node, 'current node:', current_node)
                next_node_label = current_node_labels[str(next_node)]
                
                # Get the traffic light instance
                traffic_light = self.map.nodes[str(car.next)].traffic_lights[str(previous_node_label)][str(next_node_label)]
                traffic_light_state = traffic_light.state

                # car.move(traffic light color)
                car.move(traffic_light_state)
            else:
                if car.set_off:
                    # When car has set off, when at a node queue, it is 100% of the way between previous and current nodes 
                    self.all_car_journeys[str(car.id)].append([(int(previous_node), int(current_node)), car.road_progress/100]) # current, next
                else:
                    # If it is still stuck at the origin node, given that we initialise cars on queues by giving them a random
                    # pseudo-previous_node, we can't use previous node until the car has set off as otherwise in the animation it will
                    # appear to be jumping to a random adjacent node at the very start of the replay
                    self.all_car_journeys[str(car.id)].append([(int(current_node), int(next_node)), car.road_progress/100]) # current, next


                self.reward_array[car.current] += -1
                # # get traffic light of next node [row][column]
                # traffic_light = self.map.traffic_light_instances[current_node][previous_node][next_node]
                # traffic_light_state = traffic_light.state
                self.time_in_traffic[car_count] += 1

               
                current_node_labels = self.map.nodes[str(current_node)].edge_labels
                #print('next node ', next_node, 'current node labels, ', current_node_labels)
                previous_node_label = current_node_labels[str(previous_node)]


                # if there are more nodes to go through, check the next next node
                # otherwise, this is the final destination, so move into the node and finish
                if len(car.path) > 1:
                    next_node = car.path[1]
                else:
                    car.move(True)
                    continue


                next_node_label = current_node_labels[str(next_node)]
                
                # Get the traffic light instance
                traffic_light = self.map.nodes[str(current_node)].traffic_lights[str(previous_node_label)][str(next_node_label)]
                traffic_light_state = traffic_light.state

                # car.move(traffic light color)
                car.move(traffic_light_state)

            car_count += 1
            
            #print('car id:', car.id, 'car path: ', car.path, 'road progression', car.road_progress, '\n')

        

        self.time += 1
        return self.get_local_state(), self.reward_array, finished

if __name__ =='__main__':
    test = Environment(5)
    print(test.cars)
    test.get_state()
