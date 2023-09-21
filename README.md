# Traffic Management with Reinforcement Learning

This project aims to build an environment to simulate a traffic management system and employ various reinforcement learning algorithms to optimize traffic flow. The simulation currently models a traffic network with roads, traffic lights, and intersections, allowing various algorithms to interact with the environment. 

Huge credits and thank you to William, Ishmail, and Samiul for starting and building everything so far with me.


## Project Structure

### Edge

The `Edge` class represents the roads in our traffic network. Each `Edge` has a `speed_limit`, `distance`, and `additional_costs` (which defaults to 0). The weight of the `Edge` is calculated as the time it would take for a car to travel the road's distance at the speed limit plus any additional costs.

### Graph

The `Graph` class is the core of our traffic network. It is initialized with a specified number of nodes, and it generates a graph with edges randomly distributed according to a sparsity distribution. Each edge is an instance of the `Edge` class.

The graph also includes traffic light functionality with a 3D matrix representing the locations of traffic lights in the network and an associated matrix of `Light` instances. Each intersection in the network is also labeled for easier management.

### Light

The `Light` class represents traffic lights in our traffic network. Each light has a state (0 or 1, representing green and red lights, respectively) and keeps track of the time it last changed states. Traffic lights are changable after a certain time period to simulate real-world traffic conditions.

### Node

The `Node` class represents intersections in our traffic network. Each node has a label, a list of connections (edges), and a list of queues for cars waiting at the intersection. The number of connections is also stored as the degree of the node.

### Cars

The `Car` class is another key part of the traffic network problem. Cars are initialised in random nodes and in random queues.

## Current Status and Future Work

Currently, the traffic network is generated with random edges, cars are initialised at random nodes in random queues and all features are fully functional. IQL Reinforcement learning algorithms have been applied successfully.

We decided to work on a transformer to solve this problem but quickly encountered numerous problems. These problems are seen as opportunities to work on new transformer architectures, and we hope to address several limitations. As a result, no future works have been laid out for this traffic management problem.

We welcome contributions to this project. Please feel free to reach out if you're interested in collaborating or have any questions or feedback.
