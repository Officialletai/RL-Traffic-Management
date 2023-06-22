import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge import Edge

class Graph:
    def __init__(self, num_nodes=10, sparsity_dist=[0.5, 0.5], max_weight=5, weight_dist=[0.5, 0.2, 0.1, 0.1, 0.1]):
        self.num_nodes = num_nodes
        self.sparsity_dist = sparsity_dist
        self.max_weight = max_weight
        self.weight_dist = weight_dist
        self.adjacency = self.generate()
        
    
    def generate(self):
        # Initialise an n x n adjacency matrix
        a = np.empty((self.num_nodes, self.num_nodes), dtype=object)

        # Fill in matrix ensuring diaginals are 0 (no self-loops), and that the matrix is symmetric
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                # Ensure diagonals are zero (no self-loops)
                if i==j:
                    a[i][j] = 0
                # Randomly assign 0 to cell for no connecting edge, or an Edge item with a randomly chosen weight up to a value of self.max_weight (according to the PMF given by self.weight_dist)
                else:
                    a[i][j] = random.choices([0, Edge(random.choices(np.arange(1,self.max_weight+1), weights=self.weight_dist)[0])], weights=self.sparsity_dist)[0]
                    a[j][i] = a[i][j]
            
        return a

    def print_adjacency(self):
        # Print the adjacency matrix with weights in place of Edge objects
        print_adjacency = np.empty((self.num_nodes, self.num_nodes))
        for i in range(len(self.adjacency)):
            for j in range(len(self.adjacency)):
                if isinstance(self.adjacency[i][j], Edge): 
                    print_adjacency[i][j] = self.adjacency[i][j].weight
                else:
                    print_adjacency[i][j] = self.adjacency[i][j]

        print(print_adjacency) 

    def draw(self):
        # Create a NetworkX graph object
        G = nx.Graph()

        # Add edges to the graph based on the adjacency matrix
        for i in range(len(self.adjacency)):
            for j in range(i+1, len(self.adjacency[i])):
                if self.adjacency[i][j] != 0:
                    G.add_edge(i, j)

        # Draw the graph
        pos = nx.spring_layout(G)  # Specify the layout algorithm for positioning the nodes
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()


# Create a graph instance
my_graph = Graph(10)
# Print attributes and draw graph object
print(my_graph.num_nodes)
print(my_graph.adjacency)
my_graph.draw()
print(np.all(my_graph.adjacency == my_graph.adjacency.T))
print(my_graph.print_adjacency())

