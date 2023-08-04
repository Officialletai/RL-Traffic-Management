import numpy as np
import json
import random
from edge import Edge
from light import Light

class Node:
    def __init__(self, label, connections):
        self.label = label
        self.connections = connections
        self.degree = np.count_nonzero(connections)

        self.traffic_lights = self.get_traffic_lights()

        self.queues = self.get_queues()
        self.pointers = self.get_pointers()

        self.edge_labels = self.get_edge_labels()

        # Read and store phases appropriate for a node of self.degree
        if self.degree > 1:
            filename = f'phases/node_degree_{self.degree}_phases.json'
            with open(filename, "r") as file:
                self.phases = json.load(file)

            # Store the phase numbers (i.e. 1,2,3....)
            self.phase_numbers = [i+1 for i in range(len(self.phases))]

            # Pick a random phase to initialise the node
            self.phase = random.choice(self.phase_numbers)

        else:
            # if terminal node / degree == 1: create incoming and outgoing queue respectively
            queues = {}
            queues['A'] = []
            queues['B'] = []
            self.queues = queues
            
            self.traffic_lights['A'] = {'B': Light(0)}
            self.traffic_lights['B'] = {'A': Light(0)}
            
            index_of_connected_node = int(np.nonzero(self.connections)[0])

            self.edge_labels = {self.label: 'A', str(index_of_connected_node) : 'B'}

            self.phase = 0

    def get_edge_labels(self):
        """
        Labels the edges coming into this node from a to d to help with identifying which edges are straight roads,
        left turns, right turns etc. so that phases can be translated to a setting of the lights at a junction with
        no ambiguity for the particular node/junction.
        """
        edge_labels = {}
        keys = 'ABCD'
        key_num = 0
        for i, value in enumerate(self.connections):
            if isinstance(value, Edge): 
                # edge_labels[keys[key_num]] = i
                edge_labels[str(i)] = keys[key_num]
                key_num += 1
        
        return edge_labels
    
    def get_traffic_lights(self):
        """Initialises all instances of traffic lights required for the node (degree x (degree-1) lights).
        Stored in a nested dictionary with the following format: 
        {'current_edge_1': {'next_edge_1': Light instance, 'next_edge_2': Light instance}, 'current_edge_2': ...}"""
        keys = 'ABCD'[:self.degree]
        traffic_lights = {}
        for key in keys:
            traffic_lights[key] = {keys[j]: Light(0) for j in range(self.degree) if keys[j]!=key}
        
        return traffic_lights
    
    def get_traffic_light_states(self):
        """Returns the nested dictionary of traffic light states (0 or 1 for red or green) rather than the 
        traffic light instances themselves."""
        traffic_light_states = {}
        for edge in self.traffic_lights:
            traffic_light_states[str(edge)] = {}
            for next_edge in self.traffic_lights[str(edge)]:
                traffic_light_states[str(edge)][str(next_edge)] = self.traffic_lights[str(edge)][str(next_edge)].state

        return traffic_light_states        

    def get_queues(self):
        """
        Initialises the necessary amount of queues for an intersection and stores them in a dictionary in the
        format {'edge coming into node': queue}, where 'edge coming into node' is labelles as A, B, C or D as
        defined in the get_edge_labels() method. Queues are list structures as of now. 
        """
        queues = {}
        keys = 'ABCD'

        for i in range(self.degree):
            queues[keys[i]] = []

        return queues
    
    def get_pointers(self):
        """
        Initialises the pointers for each queue at an intersection: for a node of degree n, there are n-1 pointers
        for each degree/edge, hence n x (n-1) pointers in total for a node. These are stored as a dictionary inside
        another dictionary in the following format (example):

        self.pointers = {'A': {'B': None, 'C': None}, 'B': {'A': None, 'C': None}, 'C': {'A': None, 'B': None}}

        The first item in the outer dictionary has key 'A' and thus refers to the edge labelled 'A' (as per the
        get_edge_labels() method). Cars from edge 'A' at this particular node could either want to go to edge 'B'
        or 'C' (3-way intersection). This is shown in the inner dictionary that belongs to key 'A'. Here we have 
        keys 'B' and 'C' showing cars could want to go to either of these two edges from 'A'. The values of 'B'
        and 'C' are the index of the first car wanting to go to one of these edges from the queue belonging to
        key/edge 'A' in self.queues.
        """
        pointers = {}
        keys = 'ABCD'
        
        if self.degree == 1:
            pointers['A'] = None
            pointers['B'] = None
            return pointers

        for i in range(self.degree):
            # pointers[keys[i]] = [None for next_edge in range(self.degree - 1)]
            pointers[keys[i]] = {next_edge: None for next_edge in keys[:self.degree] if next_edge != keys[i]}
        return pointers

    def update_pointers(self):
        """
        Updates the pointers to ensure the correct car is tracked. Needs to be run every time a change occurs in
        the queues (e.g. car joins or leaves the queue) as the first index of a car wanting to go to a certain edge
        could change whenever a car leaves or joins the queue. 'None' type is used for when there are no cars in a
        particular queue wanting to go to an edge. 
        """
        if self.degree == 1:
            # Track the first car in the incoming queue ('A') if there is one
            # this is sufficient because there is only one possible next edge
            self.pointers['A'] = self.queues['A'][0] if self.queues['A'] else None
            # Track the first car in the outgoing queue ('B') if there is one
            self.pointers['B'] = self.queues['B'][0] if self.queues['B'] else None
            return


        for edge in self.pointers:
            edge_destinations = [self.edge_labels[str(car.next)] for car in self.queues[edge]]
            for pointer in self.pointers[edge]:
                # print(edge_destinations.index(pointer))
                try:  
                    self.pointers[edge][pointer] = self.queues[edge][edge_destinations.index(pointer)]
                except:
                    self.pointers[edge][pointer] = None
    

