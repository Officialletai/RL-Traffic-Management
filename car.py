import random
import networkx as nx
from typing import List

import numpy as np

from environment import Graph

class Car:
    def __init__(self, car_id: int, map_: Graph, origin: int, destination: int, road_progress: int = 0, time: int = 0):
        """
        Initialises the Car object.

        Args:
            car_id (int): The car's unique identifier.
            map_ ('Graph'): The Graph object representing the road network.
            origin (int): The node where the car starts its journey.
            destination (int): The node where the car aims to reach.
            road_progress (int, optional): The progress of the car along its current road. Defaults to 0.
            time (int, optional): The time elapsed since the car started its journey. Defaults to 0.
        """
        self.id = car_id 
        self.map = map_
        self.origin = origin
        self.destination = destination
        self.path = self.path_finder()  
        self.current = origin  
        self.next = self.path[1] if self.path else None
        self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
        self.road_progress = road_progress
        self.time = time


    def get_location(self):
        """
        Get the car's current location.
        
        Returns:
            tuple: The current node and the car's progress along the road to the next node.
        """
        return self.current, self.road_progress
    

    def move(self, speed: float, light_green: bool):
        """
        Move the car along its path.
        
        Args:
            speed (float): The percentage of the road the car can traverse in one time unit.
            light_green (bool): Whether the light at the next node is green.
        """
        self.road_progress += speed
        if self.road_progress > 100:  # Ensure road progress does not exceed 100
            self.road_progress = 100

        # If the car has reached the end of the road and the light is green
        if self.road_progress >= 100 and light_green:
            self.time += 100 / speed  # Here, time is a measure of how many time units the car has been moving
            self.road_progress = 0
            self.current = self.next
            self.path.pop(0)
            if self.path:
                self.next = self.path[1] if len(self.path) > 1 else None
                self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
            else:
                self.next = None
                self.road = None
        elif not light_green and self.road_progress >= 100:
            self.time += 1  # Still increment time even if the car is not moving
        else:
            self.time += 1



    def path_finder(self) -> List[int]:
        """
        Determines the shortest path from the car's origin to its destination using Dijkstra's algorithm.

        Returns:
            List[Edge objects]: The shortest path as a list of nodes.
        """

        # Create a copy of the adjacency matrix with Edge objects replaced by their weights
        # these weights should summarise the cost of the car going through the road
        # the algorithm will attempt to minimise the cost.
        adjacency_weights = np.array([[edge.weight if edge != 0 else 0 for edge in row] for row in self.map.adjacency])

        # Convert the modified adjacency matrix to a NetworkX graph
        graph = nx.DiGraph(adjacency_weights)

        
        # Calculate the shortest path from origin to destination
        shortest_path = nx.shortest_path(graph, source=self.origin, target=self.destination, weight='weight')
        
        return shortest_path
    


if __name__ == '__main__':
    london = Graph(10) 
    start = random.randrange(0, london.num_nodes - 1)
    stop = random.randrange(0, london.num_nodes - 1)
    while stop == start:
        stop = random.randrange(0, london.num_nodes - 1)
    car_0 = Car(0, london, start, stop)
    shortest_path = car_0.path_finder()
    print(car_0.current)
    print(car_0.next)
    print(car_0.path)
    print(car_0.road)
    print(car_0.time)