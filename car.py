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
    car_0 = Car(0, london, 0, 9)
    shortest_path = car_0.path_finder()
    print(shortest_path)