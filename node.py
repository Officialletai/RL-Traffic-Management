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

        queues = {}
        keys = 'ABCD'

        for i in range(self.degree):
            queues[keys[i]] = []

        return queues
    
    def get_pointers(self):

        pointers = {}
        keys = 'ABCD'

        for i in range(self.degree):
            # pointers[keys[i]] = [None for next_edge in range(self.degree - 1)]
            pointers[keys[i]] = {next_edge: None for next_edge in keys if next_edge != keys[i]}
        return pointers

    def update_pointers(self):
        
        
        for edge in self.pointers:
            edge_destinations = [car.next for car in self.queues[edge]]
            for pointer in self.pointers[edge]:
                print(pointer)
                # 

# it has degree queues
# it has degree x (degree-1) traffic lights for fully defined case