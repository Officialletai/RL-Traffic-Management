import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge import Edge
from node import Node
from light import Light
# from controller import Controller

class Map:
    """
    Map is a class representing an undirected, weighted graph, which is our choice of representation for a real map.
    """

    def __init__(self, num_nodes=10, average_degree=2, seed=28062023):
        """
        Initialize a graph.

        num_nodes: Number of nodes in the graph.
        sparsity_dist: Probability mass function for whether an edge exists or not.
        """

        # Set the random seed
        random.seed(seed)

        # mu_distance and sigma_distance are parameters for the normal distribution from which road lengths are drawn
        self.mu_distance = 50
        self.sigma_distance = 15

        # speed_lower and speed_upper are lists of possible speed limits for different types of roads
        self.speed_lower = [20, 30]
        self.speed_upper = [60, 70]

        self.max_edge = 4

        self.num_nodes = num_nodes
        self.average_degree = average_degree
        self.sparsity_dist = [0.2, 0.8]
        self.adjacency = self.generate()
        self.weight_matrix = self.get_weight_matrix()
        self.binary_adjacency = self.get_binary_adjacency()
        self.nodes = self.generate_nodes()
        self.intersections = self.get_intersections()

        # self.traffic_light_matrix = self.traffic_light_locations()
        # self.traffic_light_instances = self.generate_traffic_light_instances()
    

    def adjacency_to_graph(self, adjacency_matrix):
        graph = nx.Graph()

        for node in range(self.num_nodes):
            graph.add_node(node)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                edge = adjacency_matrix[i][j]
                if edge != 0:
                    graph.add_edge(i, j, speed_limit=edge.speed_limit, distance=edge.distance)
        
        return graph
    

    def select_node_with_room_for_edge(self, graph, component):
        nodes = list(component)
        random.shuffle(nodes)  # Shuffle the nodes for randomness
        for node in nodes:
            if len(graph[node]) < self.max_edge:
                return node
        return None  # Return None if no suitable node is found


    def ensure_connected(self, graph):
        while not nx.is_connected(graph):
            # components == islands of nodes
            components = list(nx.connected_components(graph))
            
            # we pick two random components
            component_1 = random.choice(components)
            component_2 = random.choice([comp for comp in components if comp != component_1])

            randomness_count = 0
            while randomness_count < 5:
                node_1 = self.select_node_with_room_for_edge(graph, component_1)
                node_2 = self.select_node_with_room_for_edge(graph, component_2)
                randomness_count += 1

                if node_1 and node_2:
                    break

            if node_1 is None or node_2 is None:
                # if this error is ever raised, switch this to generating a new graph preferrably with a different sparsity distribution
                raise ValueError("Unable to ensure connectivity while obeying max_edge constraint.")
            

            # Sample a road length from a normal distribution
            random_road_length = np.random.normal(self.mu_distance, self.sigma_distance, 1)
            # min road distance is 10km
            distance = max(10, random_road_length)

            # Choose a speed limit based on the sampled road length
            if distance < (self.mu_distance - self.sigma_distance):
                speed_limit = random.choice(self.speed_lower)
            else:
                speed_limit = random.choice(self.speed_upper)
            
            graph.add_edge(node_1, node_2, speed_limit=speed_limit, distance=distance)


    def graph_to_adjacency(self, graph):

        adjacency_matrix = np.empty((self.num_nodes, self.num_nodes), dtype=object)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    adjacency_matrix[i][j] = 0

                elif graph.has_edge(i, j):
                    distance = graph[i][j]['distance']
                    speed_limit = graph[i][j]['speed_limit']
                    adjacency_matrix[i][j] = Edge(speed_limit, distance)

                else:
                    adjacency_matrix[i][j] = 0

        return adjacency_matrix
    
    def ensure_all_nodes_present(self, graph):
        for node in range(self.num_nodes):
            
            if len(graph[node]) == 0:
                # if so, node is missing, find another node to connect to
                other_nodes = list(range(self.num_nodes))
                random.shuffle(other_nodes)

                for other_node in other_nodes:
                    if len(graph[other_node]) < self.max_edge and other_node != node:
                        # connect node with random edge
                        # Sample a road length from a normal distribution
                        random_road_length = np.random.normal(self.mu_distance, self.sigma_distance, 1)
                        # min road distance is 10km
                        distance = max(10, random_road_length)

                        # Choose a speed limit based on the sampled road length
                        if distance < (self.mu_distance - self.sigma_distance):
                            speed_limit = random.choice(self.speed_lower)
                        else:
                            speed_limit = random.choice(self.speed_upper)

                        graph.add_edge(node, other_node, speed_limit=speed_limit, distance=distance)
                        break
    

    def calculate_average_degree(self, graph):
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        avg_degree = (2 * num_edges) / num_nodes
        return avg_degree
    

    def generate(self):
        """
        Generate the adjacency matrix of the graph.
        """

        total_possible_edges = (self.num_nodes * (self.num_nodes - 1)) / 2
        average_num_edges = (self.average_degree * self.num_nodes) / 2
        edge_probability = average_num_edges / total_possible_edges
        self.sparsity_dist = [1 - edge_probability, edge_probability]

        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=object)

        # Keep track of the number of edges per node
        edge_count = [0 for i in range(self.num_nodes)]

        nodes = list(range(self.num_nodes))
        patience = 0
        while patience < self.num_nodes * 4:
            i, j = random.sample(nodes, 2)
            if adjacency_matrix[i][j] != 0:
                continue
            if i != j and edge_count[i] < self.max_edge and edge_count[j] < self.max_edge:
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

                if i == 3 or j == 3:
                    print('pre 3 test')
                    print(adjacency_matrix[3])
                    print(edge_count[3])
                    

                adjacency_matrix[i][j] = edge_or_not
                adjacency_matrix[j][i] = edge_or_not

                if edge_or_not != 0:
                    print(i, j, adjacency_matrix[i][j])
                    print(j, i, adjacency_matrix[j][i])
                    edge_count[i] += 1
                    edge_count[j] += 1

                    if i == 3 or j == 3:
                        print('post 3 test')
                        print(adjacency_matrix[3])
                        print(edge_count[3])

                patience = 0
            else:
                if edge_count[i] >= self.max_edge:
                    print('full', i)
                    nodes.remove(i)
                if edge_count[j] >= self.max_edge:
                    print('full', j)
                    nodes.remove(j)
                if len(nodes) <= 3:
                    print('full', nodes)
                    break
                patience += 1

            # Break if we've reached the desired average degree
            graph = self.adjacency_to_graph(adjacency_matrix)
            if self.calculate_average_degree(graph) >= self.average_degree:
                break

        # Convert the adjacency matrix to a networkx graph for easier processing
        graph = self.adjacency_to_graph(adjacency_matrix)

        # Ensure connectivity
        self.ensure_connected(graph)
        
        # Ensure all nodes present
        self.ensure_all_nodes_present(graph)

        # Convert the graph back to an adjacency matrix
        adjacency_matrix = self.graph_to_adjacency(graph)

        avg_degree = self.calculate_average_degree(graph)
        print(f"Desired Average Degree: {self.average_degree} Actual Average Degree: {avg_degree}")

        return adjacency_matrix
    
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

        pos = nx.kamada_kawai_layout(graph)  
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()
        

    def get_intersections(self):
        num_intersections = np.sum(self.adjacency != 0, axis=1) # Finds the total intersections N at each node
        return num_intersections        
    

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
    
    def get_binary_adjacency(self):
        binary_adjacency = np.array(self.weight_matrix != 0.0)
        return binary_adjacency
    
if __name__ == '__main__':
    # Test the graph
    graph = Map(num_nodes=20, average_degree=2.2) # Creates instance of graph with 10 different nodes
    print("Nodes = ", graph.num_nodes) 
    # print("Graph adjacency = ",graph.adjacency) # Adjacency matrix 
    print(np.all(graph.adjacency == graph.adjacency.T))
    # print(graph.return_adjacency())
    graph.draw()
    # traffic_lights = graph.traffic_light_locations()
    # instance = graph.traffic_light_instances
    
    # print(graph.traffic_light_matrix)
    print(graph.weight_matrix)
    print(graph.intersections)
    
    for node_number in graph.nodes:
        print(graph.nodes[str(node_number)].traffic_lights)
