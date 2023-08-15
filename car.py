import random
import networkx as nx
from typing import List, Tuple

import numpy as np
from edge import Edge

from map import Map

# Constants
MAX_PROGRESS = 100
DEFAULT_SPEED_MULTIPLIER = (100 / 60) * 5 # Conversion to seconds from 100 percent

class Car:
    def __init__(self, car_id: int, map_: Map, origin: int, destination: int, time: int = 0):
        """
        Initializes the Car object.
        
        Args:
            car_id (int): Unique identifier for the car.
            map_ (Map): Represents the road network.
            origin (int): Starting node for the car.
            destination (int): Ending node for the car.
            time (int, optional): Time elapsed since the car started. Defaults to 0.
        """
        
        self.id = car_id 
        self.map = map_
        self.origin = origin
        self.destination = destination
        
        self.path = self.path_finder()
        self.time_path = []
        self.path_cost = self.get_path_weights()
        
        self.current = self.path[0] if self.path else None
        self.next = self.path[1] if len(self.path) > 1 else None

        self.initial_edge_choice = self.initialise_on_queue() # Choose a random edge for car to line up in start node

        self.previous = self.initial_edge_choice # origin  
        
        # road is Edge object
        self.road = self.map.adjacency[self.path[0], self.path[1]] if self.path else None
        self.road_progress = 0
        self.time = time

        self.on_edge = False # True
        self.finished = False #bool(self.current == self.destination)
        self.speed = 0
        self.set_off = False # Checks if car has set off on its journey i.e. has it left the origin node
    
    @property
    def reward(self) -> float:
        """
        Calculates the reward for the car.
        
        Reward = Time taken with no traffic - Time taken to reach destination
        
        Returns:
            float: Calculated reward.
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

        return total_time_weights - self.time
        #print('total time weights,', total_time_weights, 'self.time,', self.time)

    def initialise_on_queue(self) -> int:
        """
        Picks a random edge at the origin node for the car to queue up initially.
        
        Returns:
            int: Chosen node number.
        """
        init_node = self.map.nodes[f'{self.origin}']
        edge_labels = init_node.edge_labels
        possible_edges = list(init_node.queues.keys())
        # Do not use the edge along which it has to travel first as a potential pseudo-edge it uses for 
        # initialisation i.e. do not force any U-turns at the start

        if self.map.intersections[self.origin] == 1:
            init_node.queues['B'].append(self)
            init_node.update_pointers()

            for edge_node_number in edge_labels:
                return int(edge_node_number)


        possible_edges.remove(edge_labels[str(self.next)])
        edge_choice = random.choice(possible_edges)
        
        # Add car instance to a particular edge at origin node
        init_node.queues[str(edge_choice)].append(self)
        init_node.update_pointers()
    
        for edge_node_number in edge_labels:
            if edge_labels[edge_node_number] == edge_choice:
                return int(edge_node_number)
            #print('Check initialise_on_queue() method')


    def get_queue(self):
        """
        Returns the queue corresponding to the current or next node of the car.
        
        Returns:
            list: Queue of cars.
        """

        node_id = str(self.next if self.on_edge else self.current)
        current_node = self.map.nodes[node_id]
        
        if self.map.intersections[int(node_id)] == 1:
            return current_node.queues['A' if self.on_edge else 'B']

        # if only 1 intersection -> incoming / outgoing queue only
        current_edge_label = current_node.edge_labels[str(self.previous)]
        queue = current_node.queues[str(current_edge_label)]

        return queue
    

    def random_next_edge(self) -> str:
        """
        Selects a random edge different from the current one at the current node.
        
        Returns:
            str: Edge label.
        """

        current_node = self.map.nodes[str(self.current)]
        current_edge_label = current_node.edge_labels[str(self.previous)]

        if current_node.degree != 1:
            if len(self.path) > 1:
                return current_node.edge_labels[str(self.path[1])]

            # if self.next is none, select a random edge label
            possible_edges = list(current_node.edge_labels.values())
            # exclude the current edge label from the selection
            possible_edges.remove(current_edge_label)
            return random.choice(possible_edges)


    def if_front_of_queue(self) -> bool:
        """
        Checks if the car is at the front of its current queue.
        
        Returns:
            bool: True if car is at the front, False otherwise.
        """
        
        current_node = self.map.nodes[str(self.current)]
        current_edge_label = current_node.edge_labels[str(self.previous)]

        if current_node.degree == 1:
            return current_node.pointers['B'] == self
        
        next_edge_label = self.random_next_edge()
        return current_node.pointers[current_edge_label][next_edge_label] == self


    def pop_path(self):
        """Removes the first element from the car's path and its corresponding weight."""
        self.path.pop(0)
        if self.path_cost:
            self.path_cost.pop(0)

    def update_navigation(self):
        """Updates the car's navigation properties when it's transitioning to a new segment of its path."""
        self.on_edge = True

        self.road_progress = 0
        self.previous = self.current
        self.pop_path()

        if self.path:
            self.current = None
            self.road = self.map.adjacency[self.previous, self.next]
        else:
            self.road = None
            self.on_edge = False
            self.finished = True

        
    def get_next_node(self):
        return self.path[1] if len(self.path) > 1 else None

    
    def compute_speed(self) -> float:
        """
        Calculates the speed of the car based on the weight of its current segment.
        
        Returns:
            float: Calculated speed.
        """
        if self.path_cost:
            speed = (1 / self.path_cost[0]) * DEFAULT_SPEED_MULTIPLIER
            self.speed = speed
            return speed
        return self.speed
    

    def move(self, light_green: bool):
        """
        Moves the car based on its path, current speed, and traffic light state at its current/next node.
        
        Args:
            light_green (bool): State of the traffic light. True if it's green, False otherwise.
        """
        #print('car_id:', self.id, 'light:', light_green, 'on_edge:', self.on_edge)
        queue = self.get_queue()

        # Compute the car's speed based on the previous road's weight (cost)
        speed = self.compute_speed()

        # If a car is not on the road, it is in queue so it cannot move forward but time still passes
        # if a car is at the front of the queue, the node class will pop the car out of the queue
        if self.on_edge == False:
            if self.if_front_of_queue() and light_green == True:
                queue.remove(self)
                self.time_path.append(self.time)
                current_node = self.map.nodes[str(self.current)]
                current_node.update_pointers()
                self.update_navigation()
            else:
                self.time += 1
                return

        # Move the car along the previous road at the computed speed
        # Ensure road progress does not exceed 100
        self.road_progress = min(self.road_progress + speed, MAX_PROGRESS)
        
        if self.next == self.destination and self.road_progress == 100:
            self.finished = True
            self.time += 1
            self.time_path.append(self.time)
            return

        # If the car has reached the end of the road and the light is green
        if self.road_progress >= MAX_PROGRESS and light_green:

            # if there exists a queue, join queue, otherwise pass through
            # queue = queue if exists, otherwise returns None 

            # if queue exists and not empty
            # join queue, and now off road
            if queue and len(queue) > 0:
                queue.append(self)

                self.current = self.next
                self.next = self.get_next_node()

                # if car join queue, car no longer on edge
                self.on_edge = False

                self.map.nodes[str(self.current)].update_pointers()
                
            # if queue is empty or does not exist we go to the next road
            else:
                self.current = self.next
                self.next = self.get_next_node()
                self.update_navigation()


        # if traffic light is red and we are at end of road, we must join the correct queue
        elif self.road_progress >= MAX_PROGRESS and not light_green:
            queue.append(self)
            self.current = self.next

            self.next = self.get_next_node()
            # if car join queue, car no longer on edge
            self.on_edge = False

            self.map.nodes[str(self.current)].update_pointers()

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
            if hasattr(edge.weight, "__len__"):
                path_weights.append((edge.weight[0]))
            else:
                path_weights.append((edge.weight))
            
        return path_weights

if __name__ == '__main__':
    london = Map(3, sparsity_dist=[0.25, 0.75]) 
    start = random.randrange(0, london.num_nodes - 2)
    stop = random.randrange(1, london.num_nodes - 1)
    while stop == start:
        stop = random.randrange(0, london.num_nodes - 1)
    
    print('start,', start)
    print('stop,', stop)

    car_0 = Car(0, london, start, stop)
    shortest_path = car_0.path_finder()
    print(car_0.path)

    mapping = car_0.map.nodes[f'{car_0.current}'].edge_labels
    print('mapping',mapping)

    print('queue', car_0.get_queue())

