import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge import Edge

class Graph:
    """
    Graph is a class representing an undirected, weighted graph.
    """

    def __init__(self, num_nodes=10, sparsity_dist=[0.5, 0.5], max_weight=5, weight_dist=[0.5, 0.2, 0.1, 0.1, 0.1]):
        """
        Initialize a graph.

        num_nodes: Number of nodes in the graph.
        sparsity_dist: Probability mass function for whether an edge exists or not.
        max_weight: Maximum edge weight.
        weight_dist: Probability mass function for edge weights.
        """
        self.num_nodes = num_nodes
        self.sparsity_dist = sparsity_dist
        self.max_weight = max_weight
        self.weight_dist = weight_dist
        self.adjacency = self.generate()

    def generate(self):
        """
        Generate the adjacency matrix of the graph.
        """
        adjacency_matrix = np.empty((self.num_nodes, self.num_nodes), dtype=object)

        # Fill adjacency_matrix with weights
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if i == j:
                    adjacency_matrix[i][j] = 0
                else:
                    weight = random.choices(np.arange(1,self.max_weight+1), weights=self.weight_dist)[0]
                    edge_or_not = random.choices([0, Edge(weight)], weights=self.sparsity_dist)[0]
                    adjacency_matrix[i][j] = edge_or_not
                    adjacency_matrix[j][i] = edge_or_not

        return adjacency_matrix

    def return_adjacency(self):
        """
        Return the adjacency matrix with weights.
        """
        return_matrix = np.empty((self.num_nodes, self.num_nodes))
        for i in range(len(self.adjacency)):
            for j in range(len(self.adjacency)):
                if isinstance(self.adjacency[i][j], Edge): 
                    return_matrix[i][j] = self.adjacency[i][j].weight
                else:
                    return_matrix[i][j] = self.adjacency[i][j]

        return return_matrix

    def draw(self):
        """
        Draw the graph using matplotlib.
        """
        G = nx.Graph()

        for i in range(len(self.adjacency)):
            for j in range(i+1, len(self.adjacency[i])):
                if self.adjacency[i][j] != 0:
                    G.add_edge(i, j)

        pos = nx.spring_layout(G)  
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()



if __name__ == '__main__':
    # Test the graph
    graph = Graph(10)
    print(graph.num_nodes)
    print(graph.adjacency)
    graph.draw()
    print(np.all(graph.adjacency == graph.adjacency.T))
    graph.print_adjacency()
