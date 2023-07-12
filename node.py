import numpy as np
from edge import Edge

class Node:
    def __init__(self, label, connections, queues):
        self.label = label
        self.connections = connections
        self.queues = queues

        self.degree = np.count_nonzero(connections)
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

# it has degree queues
# it has degree x (degree-1) traffic lights for fully defined case