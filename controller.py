import json
from map import Map


class Controller:
    def __init__ (self, map_=Map):
    # def __init__ (self):
        # we use -1 as the stopping move to end the turn
        self.map = map_
        self.node_move_set = [i for i in range(self.map.num_nodes)]
        # self.node_move_set.append(-1)
        # self.move_set = self.get_phase()


        # Load move sets from files
        with open('phases/node_degree_4_phases.json', 'r') as file:
            self.four_way_move_set = json.load(file)
        with open('phases/node_degree_3_phases.json', 'r') as file:
            self.three_way_move_set = json.load(file)
        with open('phases/node_degree_2_phases.json', 'r') as file:
            self.two_way_move_set = json.load(file)
            

            # four phase 
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            # "A": {"B" : 1, "C" : 1, "D" : 0},
            # "B": {"C" : 0, "D" : 0, "A" : 0},
            # "C": {"D" : 1, "A" : 1, "B" : 0},
            # "D": {"A" : 0, "B" : 0, "C" : 0}


            # three phase
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            # "A": {"B" : 1, "C" : 0},
            # "B": {"C" : 1, "A" : 0},
            # "C": {"A" : 1, "B" : 0},
        
            # two phase
            # From : "Straight"
            # 0 = Red, 1 = Green
            # "A": {"B" : 1},
            # "B": {"A" : 1},
       
    def get_move_set(self):
        move_set = [i for i in range(self.map.num_nodes)]
        for node_number in move_set:
            if self.map.intersections[node_number] == 1:
                move_set.remove(node_number)

        # move to end turn, because technically AI can keep picking moves infintely
        # alternatively this can be used if AI picks moves == num_nodes and phase == 0 == do nothing
        #move_set.append(-1)
        return move_set
    

    def get_phase(self, num_intersections):
        if num_intersections == 4:
            return self.four_way_move_set
        elif num_intersections == 3:
            return self.three_way_move_set
        elif num_intersections == 2:
            return self.two_way_move_set
        else:
            return 0

    
    def change_traffic_lights(self, node_number, phase_number):
        # Use phase_number to generate the key for the phase
        phase_key = "phase_" + str(phase_number)
        
        # if number of intersection == 1 or phase_number == 0, skip:
        if self.map.intersections[node_number] == 1 or phase_number == 0:
            return

        # Get the appropriate phase dictionary
        phases = self.get_phase(self.map.intersections[node_number])
        
        # If the phase_key is not in the dictionary, raise an error
        if phase_key not in phases:
            raise ValueError("Invalid phase number")
        
        # Get the specific phase
        phase = phases[phase_key]
        
        # edge_label = self.map.nodes[str(node_number)].edge_labels

        keys = 'ABCD'

        # for i in keys[0:self.map.intersections[node_number]]:
        #     for j in keys[0:self.map.intersections[node_number]]:
        #         if i != j:
        #             print(self.map.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state)
        #             self.map.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state = phase[str(i)][str(j)]
        #             print('move')
        #             print(self.map.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state)

        for i in keys[0:self.map.intersections[node_number]]:
            for j in keys[0:self.map.intersections[node_number]]:
                if i!=j:
                    print(f'Node {node_number} Current Light State:')
                    print(f'{str(i)} ---> {str(j)}: ', self.map.nodes[str(node_number)].traffic_lights[str(i)][str(j)].state)
                    print('Changing Lights...')
                    self.map.nodes[str(node_number)].traffic_lights[str(i)][str(j)].state = phase[str(i)][str(j)]
                    print('New Light State:')
                    print(f'{str(i)} ---> {str(j)}: ', self.map.nodes[str(node_number)].traffic_lights[str(i)][str(j)].state)

 
if __name__ == "__main__":
    controller_test = Controller(Map())
    # print(controller_test.node_move_set)
    print(controller_test.four_way_move_set)
    print(controller_test.get_move_set())
    controller_test.change_traffic_lights(1, 1)
    controller_test.change_traffic_lights(4, 1)
    