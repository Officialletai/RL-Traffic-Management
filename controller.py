class Controller:
    def __init__ (self):
        self.four_way_move_set = self.get_four_way_phase()
        self.three_way_move_set = self.get_three_way_phase()
        self.two_way_move_set = self.get_two_way_phase()

    def get_four_way_phase(self):
        # I am assuming that all directions are relative

        phase_1 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [1, 1, 0],
            "B": [0, 0, 0],
            "C": [1, 1, 0],
            "D": [0, 0, 0]
        }

        phase_2 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [0, 0, 0],
            "B": [1, 1, 0],
            "C": [0, 0, 0],
            "D": [1, 1, 0],
        }

        phase_3 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [0, 0, 0],
            "B": [0, 0, 0],
            "C": [1, 1, 1],
            "D": [0, 0, 0]
        }

        phase_4 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [1, 1, 1],
            "B": [0, 0, 0],
            "C": [0, 0, 0],
            "D": [0, 0, 0]
        }

        phase_5 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [0, 0, 0],
            "B": [0, 0, 0],
            "C": [0, 0, 0],
            "D": [1, 1, 1],
        }
        
        phase_6 = {
            # From : "Left, Straight, Right"
            # 0 = Red, 1 = Green
            "A": [0, 0, 0],
            "B": [1, 1, 1],
            "C": [0, 0, 0],
            "D": [0, 0, 0]
        }

        four_way_intersection = [phase_1, phase_2, phase_3, phase_4, phase_5, phase_6]

        return(four_way_intersection)

    def get_three_way_phase(self):

        phase_1 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": [1, 0],
            "B": [1, 0],
            "C": [1, 0],
        }

        phase_2 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": [1, 0],
            "B": [1, 1],
            "C": [0, 0],
        }

        phase_3 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": [0, 0],
            "B": [1, 0],
            "C": [1, 1],
        }

        phase_4 = {
            # From : "Left, Right"
            # 0 = Red, 1 = Green
            "A": [1, 1],
            "B": [0, 0],
            "C": [1, 0],
        }

        three_way_intersection = [phase_1, phase_2, phase_3, phase_4]
        return three_way_intersection
    
    def get_two_way_phase(self):

        phase_1 = {
            # From : "Straight"
            # 0 = Red, 1 = Green
            "A": [1],
            "B": [1],
        }

        phase_2 = {
            # From : "Straight"
            # 0 = Red, 1 = Green
            "A": [0],
            "B": [0],
        }

        two_way_intersection = [phase_1, phase_2]
        return two_way_intersection