import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge import Edge
from node import Node
from light import Light
# from controller import Controller

class Graph:
    """
    Graph is a class representing an undirected, weighted graph.
    """

    def __init__(self, num_nodes=10, sparsity_dist=[0.65, 0.35]):
        """
        Initialize a graph.

        num_nodes: Number of nodes in the graph.
        sparsity_dist: Probability mass function for whether an edge exists or not.
        """

        # Set the random seed
        random.seed(28062023)

        # mu_distance and sigma_distance are parameters for the normal distribution from which road lengths are drawn
        self.mu_distance = 50
        self.sigma_distance = 15

        # speed_lower and speed_upper are lists of possible speed limits for different types of roads
        self.speed_lower = [20, 30]
        self.speed_upper = [60, 70]

        self.max_edge = 4

        self.num_nodes = num_nodes
        self.sparsity_dist = sparsity_dist # [no edge % chance, edge % change]
        self.adjacency = self.generate()
        self.nodes = self.generate_nodes()

        self.traffic_light_matrix = self.traffic_light_locations()
        self.traffic_light_instances = self.generate_traffic_light_instances()
        self.intersections = self.get_intersections()

        self.weight_matrix = self.get_weight_matrix()
        
        
    def generate(self):
        """
        Generate the adjacency matrix of the graph.
        """
        adjacency_matrix = np.empty((self.num_nodes, self.num_nodes), dtype=object)

        # Keep track of the number of edges per node
        edge_count = [0 for i in range(self.num_nodes)]

        # Fill adjacency_matrix with Edges (or 0s where no edge exists)
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if i == j:
                    adjacency_matrix[i][j] = 0
                else:
                    # Skip edge creation if either node has max number of nodes
                    if edge_count[i] >= self.max_edge or edge_count[j] >= self.max_edge:
                        adjacency_matrix[i][j] = 0
                        adjacency_matrix[j][i] = 0
                        continue

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

                    if edge_or_not != 0:
                        edge_count[i] += 1
                        edge_count[j] += 1

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
    
    def generate_nodes(self):
        """
        Initialises the node objects that form the graph in a dictionary
        """
        nodes = {}

        for i in range(self.num_nodes):
            nodes[str(i)] = Node(label=str(i), connections=self.adjacency[i])
        
        # Different attempts with the use of an array instead of a dictionary
        # nodes = []
        # for i in range(self.num_nodes):
        #     nodes.append(self.adjacency[i])
        return nodes
    
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
        # traffic_light_adjacency_2D = np.zeros([self.num_nodes, self.num_nodes])
        # for i, num in enumerate(num_intersections-1):
        #     traffic_light_adjacency_2D[i, binary_adjacency[i] == 1] = num
            
    
        # 3D traffic lights matrix to show connectivity
        # Current intersection (node) is the row
        # The edge that the car is travelling from (previous node) is the column
        # The edge that the car is travelling to is the depth
        # Traffic lights only exist for combinations where the previous node can go to the next node
        # There are no traffic lights at an end node (only 1 connection/intersection)
        traffic_light_adjacency_3D = np.repeat(binary_adjacency[:, :, np.newaxis], self.num_nodes, axis=2) # Repeat along the third axis
        traffic_light_adjacency_3D_axes_swapped = np.swapaxes(traffic_light_adjacency_3D, 1, 2) # Swap axes so that the depth is repeated along the columns
        traffic_light_adjacency_3D = traffic_light_adjacency_3D_axes_swapped * traffic_light_adjacency_3D # Places zeros where the 2D matrix has zeros
        for row in range(traffic_light_adjacency_3D.shape[0]):  # Emptying the diagonals
            np.fill_diagonal(traffic_light_adjacency_3D[row], 0)

        return traffic_light_adjacency_3D
    
    # Now creating a matrix of instances of all traffic lights
    def generate_traffic_light_instances(self):
        traffic_light_instances = np.empty_like(self.traffic_light_matrix, dtype=object)

        for i in range(self.traffic_light_matrix.shape[0]):
            for j in range(self.traffic_light_matrix.shape[1]):
                for k in range(self.traffic_light_matrix.shape[2]):
                    if self.traffic_light_matrix[i,j,k] != 0:
                        traffic_light_instances[i,j,k] = Light(0)
        return traffic_light_instances
    
    # Creating a matrix of instances of all traffic lights
    # 2D matrix consisting of dictionaries, many less elements compared to previous 3D matrix implementation
    # Can loop through elements using this methodology:
    """
    for i in range(a.shape[0]):         # Current node
            for j in range(a.shape[1]): # Previous node
                    for k in a[i][j]:   # Next node
    """
    def generate_traffic_light_instances_dictionary_version(self):
        traffic_light_instances = np.empty_like(self.adjacency, dtype=object)

        for i in range(self.traffic_light_matrix.shape[0]):     # Current node
            for j in range(self.traffic_light_matrix.shape[1]): # Previous node
                # if all(element == 0 for element in self.traffic_light_matrix[i,j]): # Adds a 0 element instaed of an empty dictionary in matrix[i,j]
                #     traffic_light_instances[i,j] = 0
                # else:
                    traffic_light_instances[i,j] = {}       # Initialising a dictionary
                    for k in range(self.traffic_light_matrix.shape[2]): 
                        if self.traffic_light_matrix[i,j,k] != 0:
                            traffic_light_instances[i,j][k] = Light(0) # Key is the number of the next node
        return traffic_light_instances  
    
    def get_intersections(self):
        num_intersections = np.sum(self.adjacency != 0, axis=1) # Finds the total intersections N at each node
        return num_intersections        
    
    # def change_traffic_lights(self,node_number,phase_number):
    #     cont = Controller() # Initialises controller
    #     phase = cont.get_phase(self.intersections[node_number])[phase_number]
    #     edge_label = self.nodes[str(node_number)].edge_labels
        
    #     # Changes traffic light state for all combinations of edges (All traffic lights) at a node
    #     keys = 'ABCD'
    #     for i in keys[0:self.intersections[node_number]]:
    #         for j in keys[0:self.intersections[node_number]]:
    #             if i != j:
    #                 self.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state = phase[str(i)][str(j)]
    #                 print(self.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state)
    

    def get_weight_matrix(self):
        adjacency_weights = []

        # Iterate over each row in the adjacency matrix
        for row in self.adjacency:
            # Initialize an empty list to store the values in the previous row
            new_row = []
            
            # Iterate over each item in the previous row
            for edge in row:
                # If the item is an instance of the Edge class, add its weight to the new row
                if isinstance(edge, Edge):
                    new_row.append(float(edge.weight))
                # If the item is not an instance of the Edge class (i.e., it's zero), add zero to the new row
                else:
                    new_row.append(0)
            
            # Add the new row to the new matrix
            adjacency_weights.append(new_row)

        #print(np.matrix(adjacency_weights))
        # Convert the list of lists to a numpy array
        adjacency_weights = np.array(adjacency_weights)

        return adjacency_weights


if __name__ == '__main__':
    # Test the graph
    graph = Graph(10) # Creates instance of graph with 10 different nodes
    print("Nodes = ", graph.num_nodes) 
    # print("Graph adjacency = ",graph.adjacency) # Adjacency matrix 
    print(np.all(graph.adjacency == graph.adjacency.T))
    print(graph.return_adjacency())
    graph.draw()
    traffic_lights = graph.traffic_light_locations()
    instance = graph.traffic_light_instances
    
    print(graph.traffic_light_matrix)
    print(graph.weight_matrix)
    print(graph.intersections)
