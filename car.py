class Car:
    def __init__(self, id, map, origin, destination, road_progress=0, time=0): # Not passed as arguments: path, current, next, road
        self.id = id                                                    # Car ID
        self.map = map                                                  # Graph object: pass in graph which acts as a map/satnav
        self.origin = origin                                            # Node where the car spawns
        self.destination = destination                                  # Destination node
        self.path = self.path_finder()                                  # Get shortest path from origin to destination using A* algorithm/Dijkstra's
        self.current = origin                                           # Current position is origin at the start/initialisation
        self.next = self.path[1]                                        # At the start, next node is the node at index 1 in the shortest path (index 0 is origin node)
        self.road = self.map.adjacency[self.path[0], self.path[1]]      # Edge object along which the car is currently travelling, at the start is between nodes path[0] and path[1]
        self.road_progress = road_progress                              # Store progress along road (0 when starting journey along from current node, 100% when it reaches next node)
        self.time = time                                                # Time elapsed since start of journey
    
    def path_finder(self):
        # Implement shortest path finding algorithm
        return None