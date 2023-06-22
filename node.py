import numpy as np

class Node:
    def __init__(self, label, connections, queues):
        self.label = label
        self.connections = connections
        self.degree = np.count_nonzero(connections)
        self.queues = queues
        # self.lights = 

        

# it has degree queues
# it has degree x (degree-1) traffic lights for fully defined case