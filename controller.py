from environment import Graph


class Controller:
    def __init__ (environment_object, map_=Graph):
    # def __init__ (self):
        # we use -1 as the stopping move to end the turn
        environment_object.node_move_set = [i for i in range(map_.num_nodes)]
        environment_object.node_move_set.append(-1)
        # self.move_set = self.get_phase()
        environment_object.four_way_move_set = environment_object.get_four_way_phase()
        environment_object.three_way_move_set = environment_object.get_three_way_phase()
        environment_object.two_way_move_set = environment_object.get_two_way_phase()

    def get_four_way_phase(self):
        # I am assuming that all directions are relative

        phase_1 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 1, "C" : 1, "D" : 0},
            "B": {"C" : 0, "D" : 0, "A" : 0},
            "C": {"D" : 1, "A" : 1, "B" : 0},
            "D": {"A" : 0, "B" : 0, "C" : 0}
        }
        phase_2 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 0, "C" : 0, "D" : 0},
            "B": {"C" : 1, "D" : 1, "A" : 0},
            "C": {"D" : 0, "A" : 0, "B" : 0},
            "D": {"A" : 1, "B" : 1, "C" : 0}
        }
        
        phase_3 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 0, "C" : 0, "D" : 0},
            "B": {"C" : 0, "D" : 0, "A" : 0},
            "C": {"D" : 1, "A" : 1, "B" : 1},
            "D": {"A" : 0, "B" : 0, "C" : 0}
        }
        phase_4 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 1, "C" : 1, "D" : 1},
            "B": {"C" : 0, "D" : 0, "A" : 0},
            "C": {"D" : 0, "A" : 0, "B" : 0},
            "D": {"A" : 0, "B" : 0, "C" : 0}
        }
        phase_5 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 0, "C" : 0, "D" : 0},
            "B": {"C" : 0, "D" : 0, "A" : 0},
            "C": {"D" : 0, "A" : 0, "B" : 0},
            "D": {"A" : 1, "B" : 1, "C" : 1}
        }
        
        phase_6 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 0, "C" : 0, "D" : 0},
            "B": {"C" : 1, "D" : 1, "A" : 1},
            "C": {"D" : 0, "A" : 0, "B" : 0},
            "D": {"A" : 0, "B" : 0, "C" : 0}
        }
        
        four_way_intersection = [phase_1, phase_2, phase_3, phase_4, phase_5, phase_6]

        return(four_way_intersection)

    def get_three_way_phase(self):

        phase_1 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 1, "C" : 0},
            "B": {"C" : 1, "A" : 0},
            "C": {"A" : 1, "B" : 0},
        }

        phase_2 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 1, "C" : 0},
            "B": {"C" : 1, "A" : 1},
            "C": {"A" : 0, "B" : 0},
        }

        phase_3 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 0, "C": 0},
            "B": {"C" : 1, "A": 0},
            "C": {"A" : 1, "B": 1},
        }

        phase_4 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": {"B" : 1,"C" : 1},
            "B": {"C" : 0,"A" : 0},
            "C": {"A" : 1,"B" : 0},
        }

        three_way_intersection = [phase_1, phase_2, phase_3, phase_4]
        return three_way_intersection
    
    def get_two_way_phase(self):

        phase_1 = {
            # From : "Straight"
            # 0 = Red, 1 = Green
            "A": {"B" : 1},
            "B": {"A" : 1},
        }

        phase_2 = {
            # From : "Straight"
            # 0 = Red, 1 = Green
            "A": {"B" : 0},
            "B": {"A" : 0},
        }

        two_way_intersection = [phase_1, phase_2]
        return two_way_intersection
    
    def get_phase(self, num_intersections):
        if num_intersections == 4:
            return self.get_four_way_phase()
        elif num_intersections == 3:
            return self.get_three_way_phase()
        elif num_intersections == 2:
            return self.get_two_way_phase()
        else:
            return 0
    
    def change_traffic_lights(self,node_number,phase_number,environment_object=Graph):
        phase = self.get_phase(environment_object.intersections[node_number])[phase_number]
        edge_label = environment_object.nodes[str(node_number)].edge_labels
    
        # Changes traffic light state for all combinations of edges (All traffic lights) at a node
        keys = 'ABCD'
        for i in keys[0:environment_object.intersections[node_number]]:
            for j in keys[0:environment_object.intersections[node_number]]:
                if i != j:
                    environment_object.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state = phase[str(i)][str(j)]
                    print(environment_object.traffic_light_instances[node_number][edge_label[str(i)]][edge_label[str(j)]].state)
    

 
if __name__ == "__main__":
    controller_test = Controller(Graph())
    # print(controller_test.node_move_set)
    print(controller_test.four_way_move_set)