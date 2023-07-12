from environment import Graph
from car import Car

import numpy as np
import random


"""
Run a simulation (without agent for now):

1. Initialise Environment
2. Initialise Cars 
3. Add Cars to queues at Nodes

for t in range(T) (or until completion of episode i.e. every car at its destination):
    set lights everywhere (randomly if without agent)
    move cars from queues by checking the light
    check cumulative distance left to destination from all cars
"""
class Simulation:
    def __init__(self):
        self.environment = Graph()
        self.cars = self.initialise_cars(num_cars=10)
        self.time = 0
        self.score = 0


    def initialise_cars(self, num_cars):
        cars_list = []
        for index in range(num_cars):
            start = random.randrange(0, self.environment.num_nodes - 1)
            stop = random.randrange(0, self.environment.num_nodes - 1)
            car = Car(index, self.environment, start, stop)
            cars_list.append(car)

        return cars_list


    def main():
        pass

if __name__ =='__main__':
    test = Simulation()
    print(test.cars)