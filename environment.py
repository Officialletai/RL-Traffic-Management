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
            car = Car(index, self.map, start, stop)
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
                edge_matrix[car.previous][car.next] += 1
            else:
                queue_matrix[car.current][car.next] += 1
        
        # traffic light matrix
        max_degree = max([self.map.nodes[str(node)].degree for node in self.map.nodes])
        traffic_light_matrix = np.zeros((self.map.num_nodes, max_degree*(max_degree-1)))
        
        for node in range(self.map.num_nodes):
            nodal_traffic_lights = self.map.nodes[str(node)].traffic_lights
            node_legal_journeys_counter = 0
            for current_edge in nodal_traffic_lights:
                for next_edge in nodal_traffic_lights[current_edge]:
                    traffic_light_matrix[node, node_legal_journeys_counter] = nodal_traffic_lights[current_edge][next_edge].state
                    node_legal_journeys_counter += 1
        
        return normalised_weight_matrix, queue_matrix, edge_matrix, traffic_light_matrix


    def step(self, actions):

        for node, phase in actions:
            self.controller.change_traffic_lights(node, phase)
        
        for car in self.cars[:]:
            # if car can move (not arrived at destination)
            if car.current:
                current_node = car.current
                previous_node = car.previous
                next_node = car.next


                # if the car is at the end of its destination and the node is a dead end
                # move as though there was a green light (no traffic light exists there)
                if not next_node and self.map.intersections[current_node] == 1:
                    car.move(light_green=True)
                    reward = car.calculate_reward()
                    self.score += reward
                    self.cars.remove(car)

                else:
                    # # get traffic light of next node [row][column]
                    # traffic_light = self.map.traffic_light_instances[current_node][previous_node][next_node]
                    # traffic_light_state = traffic_light.state

                    current_node_labels = self.map.nodes[current_node].edge_labels
                    previous_node_label = current_node_labels[previous_node]
                    next_node_label = current_node_labels[next_node]
                    
                    # Get the traffic light instance
                    traffic_light = self.map.nodes[str(current_node)].traffic_lights[str(previous_node_label)][str(next_node_label)]
                    traffic_light_state = traffic_light.state

                    # car.move(traffic light color)
                    car.move(traffic_light_state)

        reward = None 
        finished = None
                     
        return self.get_state(), reward, finished


if __name__ =='__main__':
    test = Environment()
    print(test.cars)
    test.get_state()