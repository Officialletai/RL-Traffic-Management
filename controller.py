# from environment import Graph


class Controller:
    # def __init__ (self, map_=Graph):
    def __init__ (self):
        # we use -1 as the stopping move to end the turn
        # self.node_move_set = [i for i in range(map_.num_nodes)]
        # self.node_move_set.append(-1)
        # self.move_set = self.get_phase()
        self.four_way_move_set = self.get_four_way_phase()
        self.three_way_move_set = self.get_three_way_phase()
        self.two_way_move_set = self.get_two_way_phase()

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
    

 
if __name__ == "__main__":
    # controller_test = Controller(Graph())
    controller_test = Controller()
    # print(controller_test.node_move_set)
    print(controller_test.four_way_move_set)