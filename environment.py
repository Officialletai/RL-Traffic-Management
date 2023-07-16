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

        return self.get_state()

    def get_state(self):
        weight_matrix = self.map.weight_matrix
        max_weight = weight_matrix.max()
        normalised_weight_matrix = weight_matrix / max_weight
        


    def step(self, actions):

        for node, phase in actions:
            self.controller.change_traffic_lights(node, phase)
        
        for car in self.cars:
            # if car can move (not arrived at destination)
            if car.current:
                current_node = car.current
                previous_node = car.previous
                next_node = car.next
                # get traffic light of next node [row][column]
                traffic_light = self.map.traffic_light_instances[current_node][previous_node][next_node]
                traffic_light_state = traffic_light.state

                # car.move(traffic light color)
                car.move(traffic_light_state)
            pass

        reward = None 
        finished = None
                     
        return self.get_state(), reward, finished





if __name__ =='__main__':
    test = Environment()
    print(test.cars)
    test.get_state()