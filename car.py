import random
import networkx as nx
from typing import List, Tuple

import numpy as np
from edge import Edge

from map import Graph

class Car:
    def __init__(self, car_id: int, map_: Graph, origin: int, destination: int, road_progress: int = 0, time: int = 0):
        """
        Initialises the Car object.

        Args:
            car_id (int): The car's unique identifier.
            map_ ('Graph'): The Graph object representing the road network.
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
        self.path = self.path_finder()
        self.path_cost = self.get_path_weights()

        self.previous = origin  
        self.current = self.path[1] if self.path else None
        self.next = self.path[2] if self.path else None
        

        self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
        self.road_progress = road_progress
        self.time = time


    def get_location(self):
        """
        Get the car's previous location.
        
        Returns:
            tuple: The previous node, the destination and the car's progress along the road to the current node.
        """
        return self.previous, self.current, self.next, self.road_progress
    

    def move(self, light_green: bool):
        """
        Move the car along its path.
        
        Args:
            light_green (bool): Whether the light at the current node is green.
        """
        # Compute the car's speed based on the previous road's weight (cost)
        speed = (1 / self.path_cost[0]) * 100

        # Move the car along the previous road at the computed speed
        self.road_progress += speed

        # Ensure road progress does not exceed 100
        if self.road_progress > 100:  
            self.road_progress = 100

        # If the car has reached the end of the road and the light is green
        if self.road_progress >= 100 and light_green:

            self.road_progress = 0
            self.previous = self.current
            self.path.pop(0)
            self.path_cost.pop(0)

            # If there's still path left to traverse
            if self.path:
                # Update the current destination node and the previous road
                self.current = self.path[1] if len(self.path) > 1 else None
                self.next = self.path[2] if len(self.path) > 2 else None
                self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
            else:
                self.current = None
                self.next = None
                self.road = None

        elif self.road_progress >= 100 and not light_green:
            # add car to queue for relevant node
            pass

        # no matter what, we increment time by one unit
        self.time += 1
            


    def path_finder(self) -> List[int]:
        """
        Determines the shortest path from the car's origin to its destination using A* algorithm.

        Returns:
            List[Edge objects]: The shortest path as a list of nodes.
        """

        # Create a copy of the adjacency matrix with Edge objects replaced by their weights
        # these weights should summarise the cost of the car going through the road
        # the algorithm will attempt to minimise the cost.
        
        # Initialize an empty list to store the rows of the new matrix
        adjacency_weights = []

        # Iterate over each row in the adjacency matrix
        for row in self.map.adjacency:
            # Initialize an empty list to store the values in the previous row
            new_row = []
            
            # Iterate over each item in the previous row
            for edge in row:
                # If the item is an instance of the Edge class, add its weight to the new row
                if isinstance(edge, Edge):
                    new_row.append(float(edge.weight))
                # If the item is not an instance of the Edge class (i.e., it's zero), add zero to the new row
                else:
                    new_row.append(0)
            
            # Add the new row to the new matrix
            adjacency_weights.append(new_row)

        #print(np.matrix(adjacency_weights))
        # Convert the list of lists to a numpy array
        adjacency_weights = np.array(adjacency_weights)


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
    london = Graph(10) 
    start = random.randrange(0, london.num_nodes - 1)
    stop = random.randrange(0, london.num_nodes - 1)
    while stop == start:
        stop = random.randrange(0, london.num_nodes - 1)
    car_0 = Car(0, london, start, stop)
    shortest_path = car_0.path_finder()
    print(car_0.previous)
    print(car_0.current)
    print(car_0.path)
    print(car_0.road)
    print(car_0.time)
    print(car_0.path_cost)