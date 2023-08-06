from controller import Controller
from map import Map
from car import Car

import numpy as np
import random


class Environment:
    def __init__(self):
        self.map = Map()
        self.cars = self.initialise_cars(num_cars=10)
        self.controller = Controller(self.map)

        self.time = 0
        self.score = 0


    def initialise_cars(self, num_cars):
        cars_list = []
        for index in range(num_cars):
            start = random.randrange(0, self.map.num_nodes)
            stop = random.randrange(0, self.map.num_nodes)
            # if stop == start:
            #     stop += 1
            while stop == start:
                stop = random.randrange(0, self.map.num_nodes)
            
            car = Car(index, self.map, start, stop)

            # car.initialise_on_queue()

            cars_list.append(car)
            

        return cars_list


    def reset(self):
        self.map = Map()
        self.cars = self.initialise_cars(num_cars=10)
        self.controller = Controller(self.map)

        self.time = 0
        self.score = 0

        return self.get_state()

    def get_state(self):
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
                print('car id:', car.id, 'car previous:', car.previous, 'car next:', car.next)
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
    
    def get_local_state(self, agent_node):
        node = self.map.nodes[str(agent_node)]

        # Initialise separate empty state spaces
        local_queue_matrix = np.zeros((node.degree, node.degree))
        local_edge_vector = np.zeros((1, node.degree))
        local_traffic_light_phase = np.zeros((1, node.degree*(node.degree-1)))

        for car in self.cars:
            # If a car is at its destination, no longer include it in the state
            if car.finished == True:
                continue

            # In local terms, aka in A B C D, get the data of the car
            label_previous = node.edge_labels.get(str(car.previous))
            label_next = node.edge_labels.get(str(car.next))

            # If the label exists, add to edge vector
            if label_previous:
                # Convert to A,B,C,D to 0,1,2,3 for matrix index
                previous_pos = ord(label_previous) - ord('A')
                next_pos = ord(label_next) - ord('A')

            # If the car is on an edge and it is coming towards the node
            if car.on_edge == True and car.current == agent_node:
                # Add to appropriate position in edge vector
                local_edge_vector[previous_pos] += 1

            # Else the car must be in a queue, so add to appropriate position in queue matrix
            else:
                local_queue_matrix[previous_pos, next_pos] += 1

        # Get the traffic light phases in 1s and 0s
        #
        local_traffic_light_phase = [1]
        #
        #

        # Normalise the state observations
        n_local_queue_matrix = local_queue_matrix / np.linalg.norm(local_queue_matrix, axis=1, keepdims=True)
        n_local_queue_matrix = local_queue_matrix / np.linalg.norm(local_queue_matrix, axis=1, keepdims=True)
        
        # Create the state by combining and flattening all observations together into one array
        state= [n_local_queue_matrix.flatten(), n_local_queue_matrix.flatten(),local_traffic_light_phase]

        return state
    
    def step(self, actions):
        
        finished = True

        for node, phase in actions:
            self.controller.change_traffic_lights(node, phase)
        
        for car in self.cars[:]:
            # if car can move (not arrived at destination)
            """
            Check all car nodes
            current could be none if on edge
            previous is origina / last node
            next is next node unless you have finished

            if on edge, 
            """
            if car.finished:
                print(f'car {car.id} has finished')
                reward = car.calculate_reward()
                self.score += reward
                self.cars.remove(car)
                continue
            else:
                finished = False

            current_node = car.current
            previous_node = car.previous
            next_node = car.next

            print('car id:', car.id, 'car path: ', car.path, 'road progression', car.road_progress)
            
            # if car is not edge, it must be in queue
            # if car is on road and has more intersections to get through, check them
            # if no more intersections after the coming node, it is moving towards destination
            # it will just join any queue and end there.
            if car.on_edge:
                
                
                current_node_labels = self.map.nodes[str(next_node)].edge_labels
                
                print('previous node ', previous_node, 'current node labels, ', current_node_labels)
                
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
                # # get traffic light of next node [row][column]
                # traffic_light = self.map.traffic_light_instances[current_node][previous_node][next_node]
                # traffic_light_state = traffic_light.state

               
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

            print('car id:', car.id, 'car path: ', car.path, 'road progression', car.road_progress, '\n')


        return self.get_state(), self.score, finished


if __name__ =='__main__':
    test = Environment()
    print(test.cars)
    test.get_state()