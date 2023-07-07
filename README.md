# Traffic Management with Reinforcement Learning

This project aims to build an environment to simulate a traffic management system and employ various reinforcement learning algorithms to optimize traffic flow. The simulation currently models a traffic network with roads, traffic lights, and intersections, allowing various algorithms to interact with the environment.

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

## Current Status and Future Work

Currently, the traffic network is generated with random edges, and traffic light functionality is partially implemented. 

Our future work will focus on:

1. Finalizing traffic light functionality: We will complete the simulation of traffic light behavior, including state changes and interactions with vehicles.
2. Adding vehicle objects: We plan to add car objects to simulate traffic flow in the network.
3. Implementing reinforcement learning algorithms: The main goal of this project is to apply reinforcement learning algorithms to optimize traffic flow. A rough back of envelope calculate suggests that the action state space comes close 9 figures. We attempt to apply various novel and experimental reinforcement learning algorithms, building on top of the following: Double DQN, Simulated annealing / evolutionary policy based RL, among the actor critic algorithms out there. 
4. Visualizing the traffic network: We will improve the current network visualization, which will be crucial in understanding and debugging the behavior of our reinforcement learning agents.
5. Dynamic car behaviour: We will include dynamic car behaviour with random probability in conjunction with reinforcement learning car agents to simulate typical car behaviour. 

We welcome contributions to this project. Please feel free to reach out if you're interested in collaborating or have any questions or feedback.
