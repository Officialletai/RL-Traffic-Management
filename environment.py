from controller import Controller
from map import Graph
from car import Car

import numpy as np
import random


class Environment:
    def __init__(self):
        self.map = Graph()
        self.cars = self.initialise_cars(num_cars=10)
        self.controller = Controller(self.map)

        self.time = 0
        self.score = 0


    def initialise_cars(self, num_cars):
        cars_list = []
        for index in range(num_cars):
            start = random.randrange(0, self.map.num_nodes - 1)
            stop = random.randrange(0, self.map.num_nodes - 1)
            car = Car(index, self.map, start, stop)
            cars_list.append(car)

        return cars_list


    def reset(self):
        self.map = Graph()
        self.cars = self.initialise_cars(num_cars=10)
        self.controller = Controller(self.map)
        
        self.time = 0
        self.score = 0


    def step(self, actions):

        for node, phase in actions:
            self.controller.change_traffic_lights(node, phase)
        
        for car in self.cars:
            next_node = car.next
            # get traffic light of next node
            # car.move(traffic light color)

            pass

        reward = None 
        finished = None
        new_state = (self.cars, self.map.traffic_light_instances)
                     
        return new_state, reward, finished





if __name__ =='__main__':
    test = Environment()
    print(test.cars)