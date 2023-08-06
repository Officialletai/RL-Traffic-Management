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
                self.score += car.reward
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