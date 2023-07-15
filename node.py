import numpy as np
from edge import Edge

class Node:
    def __init__(self, label, connections):
        self.label = label
        self.connections = connections
        self.degree = np.count_nonzero(connections)

        self.queues = self.get_queues()
        self.pointers = self.get_pointers()

        self.edge_labels = self.get_edge_labels()

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
                edge_labels[keys[key_num]] = i
                key_num += 1
        
        return edge_labels
    
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
        for edge in self.pointers:
            edge_destinations = [car.next for car in self.queues[edge]]
            for pointer in self.pointers[edge]:
                try:
                    self.pointers[edge][pointer] = edge_destinations.index(pointer)
                except:
                    self.pointers[edge][pointer] = None

# it has degree queues
# it has degree x (degree-1) traffic lights for fully defined case