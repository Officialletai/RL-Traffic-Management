import random
import networkx as nx
from typing import List, Tuple

import numpy as np
from edge import Edge

from map import Map

class Car:
    def __init__(self, car_id: int, map_: Map, origin: int, destination: int, road_progress: int = 0, time: int = 0):
        """
        Initialises the Car object.

        Args:
            car_id (int): The car's unique identifier.
            map_ ('Map'): The Map object representing the road network.
            origin (int): The node where the car starts its journey.
            destination (int): The node where the car aims to reach.
            road_progress (int, optional): The progress of the car along its previous road. Defaults to 0.
            time (int, optional): The time elapsed since the car started its journey. Defaults to 0.
        """
        
        
        # Set the random seed
        random.seed(28062023)
        
        self.id = car_id 
        self.map = map_
        self.origin = origin
        self.destination = destination
        
        # self.adjacency_weights = []

        self.path = self.path_finder()
        self.path_cost = self.get_path_weights()
        
        self.initial_edge_choice = self.initialise_on_queue() # Choose a random edge for car to line up in start node

        self.previous = self.initial_edge_choice # origin  
        self.current = self.path[0] if self.path else None
        self.next = self.path[1] # if len(self.path) > 2 else None
        
        # road is Edge object
        self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
        self.road_progress = road_progress
        self.time = time
        self.reward = 0

        self.on_edge = False # True
        self.finished = bool(self.current == self.destination)

    def get_location(self):
        """
        Get the car's previous location.
        
        Returns:
            tuple: The previous node, the destination and the car's progress along the road to the current node.
        """
        return self.previous, self.current, self.next, self.road_progress
    
    def calculate_reward(self):
        """
        The reward for each individual car is equal to the (time that the car took to reach it's destination) minus
        (the time the car would've taken with no traffic)
        """

        # Initialize an empty list to store the time weights
        total_time_weights = 0
        
        # Iterate over each pair of nodes in the shortest path
        for i in range(len(self.path) - 1):
            # Get the previous node and the current node in the path
            node1, node2 = self.path[i], self.path[i + 1]
            
            # Get the Edge object connecting the two nodes
            edge = self.map.adjacency[node1][node2]
            
            # Add the edge weight to the list
            total_time_weights += (edge.time_weight[0])

        self.reward = total_time_weights - self.time
    
    def initialise_on_queue(self):
        """Looks at the origin node edges and picks one randomly to queue up there at the start of its journey"""
        init_node = self.map.nodes[f'{self.origin}']
        possible_edges = list(init_node.queues.keys())
        edge_choice = random.choice(possible_edges)
        # Add car instance to a particular edge at origin node
        init_node.queues[str(edge_choice)].append(self)

        for edge_node_number in init_node.edge_labels:
            if init_node.edge_labels[edge_node_number] == edge_choice:
                return edge_node_number
        print('Check initialise_on_queue() method')

    def get_queue(self):
            # if only 1 intersection -> no traffic light -> no queue
            if not self.current or self.map.intersections[self.current] == 1:
                return None
            
            #####
            current_node = self.map.nodes[str(self.current)]
            current_edge_label = current_node.edge_labels[str(self.previous)]
            queue = current_node.queues[str(current_edge_label)]

            return queue
            #####

            # # node = previous node / from 
            # node = self.map.nodes[f'{self.previous}']
            # # get edge label of current node 
            # mapping = node.edge_labels
            # # get translation of next node -> a/b/c/d
            # translation = list(mapping.keys())[list(mapping.values()).index(self.current)]
            # return node.queues[translation]
    

    def update_navigation(self):
        self.on_edge = True

        self.road_progress = 0
        self.previous = self.current
        self.path.pop(0)
        self.path_cost.pop(0)

        #######
        if self.path:
            self.current = None
            self.next = self.path[0]
            self.road = self.map.adjacency[self.previous, self.next]
        else:
            self.current = self.current
            self.next = None
            self.road = None
            self.on_edge = False
            self.finished = True
        #######

        # if self.path:
        #     self.current = self.path[1] if len(self.path) > 1 else None
        #     self.next = self.path[2] if len(self.path) > 2 else None
        #     self.road = self.map.adjacency[self.path[0], self.path[1]] if len(self.path) > 1 else None
        # else:
        #     self.current = None
        #     self.next = None
        #     self.road = None


    def move(self, light_green: bool):
        """
        Move the car along its path.
        
        Args:
            light_green (bool): Whether the light at the current node is green.
        """

        # If a car is not on the road, it is in queue so it cannot move forward but time still passes
        # if a car is at the front of the queue, the node class will pop the car out of the queue
        if self.on_edge == False:
            self.time += 1
            return 
        
        # Compute the car's speed based on the previous road's weight (cost)
        speed = (1 / self.path_cost[0]) * 100

        # Move the car along the previous road at the computed speed
        if self.road_progress < 100:
            self.road_progress += speed

        # Ensure road progress does not exceed 100
        if self.road_progress > 100:  
            self.road_progress = 100

        # If the car has reached the end of the road and the light is green
        if self.road_progress >= 100 and light_green:

            # if there exists a queue, join queue, otherwise pass through
            # queue = queue if exists, otherwise returns None 
            queue = self.get_queue()

            # if queue exists and not empty, join queue, and now off road
            if queue and len(queue) > 0:
                queue.append(self)
                
                # if car join queue, car no longer on edge
                self.on_edge = False

            # if queue is empty or does not exist we go to the next road
            else:
                self.update_navigation()


        # if traffic light is red and we are at end of road, we must join the correct queue
        elif self.road_progress >= 100 and not light_green:
            queue = self.get_queue()
            queue.append(self)

            # if car join queue, car no longer on edge
            self.on_edge = False

        # no matter what, we increment time by one unit
        self.time += 1
            


    def path_finder(self) -> List[int]:
        """
        Determines the shortest path from the car's origin to its destination using A* algorithm.

        Returns:
            List[Edge objects]: The shortest path as a list of nodes.
        """
        adjacency_weights = self.map.weight_matrix


        # Convert the modified adjacency matrix to a NetworkX graph
        graph = nx.DiGraph(adjacency_weights)

        
        # Calculate the shortest path from origin to destination via A* 
        shortest_path = nx.astar_path(graph, source=self.origin, target=self.destination, weight='weight')
        
        return shortest_path
    

    def get_path_weights(self) -> List[Tuple[Edge, float]]:
        """
        Given the shortest path as a list of nodes, return a list of tuples. Each tuple contains an edge (i.e., 
        an Edge object) and its corresponding weight.
        """
        
        shortest_path = self.path

        # Initialize an empty list to store the weights
        path_weights = []
        
        # Iterate over each pair of nodes in the shortest path
        for i in range(len(shortest_path) - 1):
            # Get the previous node and the current node in the path
            node1, node2 = shortest_path[i], shortest_path[i + 1]
            
            # Get the Edge object connecting the two nodes
            edge = self.map.adjacency[node1][node2]
            
            # Add the edge weight to the list
            path_weights.append((edge.weight[0]))
            
        return path_weights

if __name__ == '__main__':
    london = Map(10) 
    start = random.randrange(0, london.num_nodes - 1)
    stop = random.randrange(5, london.num_nodes - 1)
    while stop == start:
        stop = random.randrange(0, london.num_nodes - 1)
    car_0 = Car(0, london, start, stop)
    shortest_path = car_0.path_finder()
    print(car_0.path)

    mapping = car_0.map.nodes[f'{car_0.current}'].edge_labels
    print('mapping',mapping)

    print('queue', car_0.get_queue())
    car_0.move(True)
    print(car_0.road_progress)
    print(car_0.previous)
    print(car_0.current)
    print(car_0.next)
    print(car_0.path)

    print(car_0.get_queue())
    car_0.move(True)
    print(car_0.road_progress)
    print(car_0.previous)
    print(car_0.current)
    print(car_0.next)
    print(car_0.path)

    print(car_0.get_queue())
