import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge import Edge

class Graph:
    """
    Graph is a class representing an undirected, weighted graph.
    """

    def __init__(self, num_nodes=10, sparsity_dist=[0.65, 0.35], max_weight=5, weight_dist=[0.5, 0.2, 0.1, 0.1, 0.1]):
        """
        Initialize a graph.

        num_nodes: Number of nodes in the graph.
        sparsity_dist: Probability mass function for whether an edge exists or not.
        max_weight: Not used in this implementation.
        weight_dist: Not used in this implementation.
        """

        # Set the random seed
        random.seed(28062023)

        # mu_distance and sigma_distance are parameters for the normal distribution from which road lengths are drawn
        self.mu_distance = 50
        self.sigma_distance = 15

        # speed_lower and speed_upper are lists of possible speed limits for different types of roads
        self.speed_lower = [20, 30]
        self.speed_upper = [60, 70]

        self.num_nodes = num_nodes
        self.sparsity_dist = sparsity_dist # [no edge % chance, edge % change]
        self.adjacency = self.generate()
    def generate(self):
        """
        Generate the adjacency matrix of the graph.
        """
        adjacency_matrix = np.empty((self.num_nodes, self.num_nodes), dtype=object)

        # Fill adjacency_matrix with Edges (or 0s where no edge exists)
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if i == j:
                    adjacency_matrix[i][j] = 0
                else:
                    # Sample a road length from a normal distribution
                    random_road_length = np.random.normal(self.mu_distance, self.sigma_distance, 1)
                    # min road distance is 10km
                    distance = max(10, random_road_length)

                    # Choose a speed limit based on the sampled road length
                    if distance < (self.mu_distance - self.sigma_distance):
                        speed_limit = random.choice(self.speed_lower)
                    else:
                        speed_limit = random.choice(self.speed_upper)

                    # Decide whether to create an Edge or not based on the sparsity distribution
                    edge_or_not = random.choices([0, Edge(speed_limit, distance)], weights=self.sparsity_dist)[0]
                    adjacency_matrix[i][j] = edge_or_not
                    adjacency_matrix[j][i] = edge_or_not

        return adjacency_matrix

    def return_adjacency(self):
        """
        Return the adjacency matrix with weights (i.e., speed limits).
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
        graph = nx.Graph()

        for i in range(len(self.adjacency)):
            for j in range(i+1, len(self.adjacency[i])):
                if self.adjacency[i][j] != 0:
                    graph.add_edge(i, j)

        pos = nx.spring_layout(graph)  
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()
        
    # Creates an adjacency matrix where in the third dimension indicates the number of traffic lights located on an edge.
    # Each row indicates a node, where the edge location at the row represent the traffic lights at the node
    def traffic_light_locations(self): 
        num_intersections = np.sum(self.adjacency != 0, axis=1) # Finds the total intersections N at each node
        binary_adjacency = (self.adjacency != 0).astype(int) # Creates a binary adjacency matrix
        
        # Adjacency matrix of traffic lights. Elements show number of traffic lights at an edge
        traffic_light_adjacency_2D = np.zeros([self.num_nodes, self.num_nodes])
        for i, num in enumerate(num_intersections-1):
            traffic_light_adjacency_2D[i, binary_adjacency[i] == 1] = num
            
        # Adjacency matrix of traffic lights, however 3 dimensional.
        traffic_light_adjacency_3D = np.zeros((self.num_nodes, self.num_nodes, np.max(num_intersections-1))) # Third dimension refers to the traffic lights
        for i, num in enumerate(num_intersections - 1):
            traffic_light_adjacency_3D[i, :, :num] = binary_adjacency[i, :, np.newaxis].astype(int)
        
        
        # 3D matrix, where only the diagonal of the column and depth rows are filled in. 
        # traffic_light_adjacency_3D_diagonal = np.zeros((self.num_nodes, self.num_nodes, self.num_nodes))
        # for row in range(self.num_nodes):
        #     np.fill_diagonal(traffic_light_adjacency_3D_diagonal[row], traffic_light_adjacency_2D[row])
        return traffic_light_adjacency_3D
    

if __name__ == '__main__':
    # Test the graph
    graph = Graph(10) # Creates instance of graph with 10 different nodes
    print("Nodes = ", graph.num_nodes) 
    # print("Graph adjacency = ",graph.adjacency) # Adjacency matrix 
    print(np.all(graph.adjacency == graph.adjacency.T))
    print(graph.return_adjacency())
    graph.draw()
    graph.traffic_light_locations()

    
