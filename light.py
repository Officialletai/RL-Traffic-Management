# State space for individual traffic light = R/G, or in binary 0/1

class Light:
    def __init__(self):
        self.state = 0

    def change_state(self):
        if self.state == 0:
            self.state = 1
        else:
            self.state = 0
        
        # The logic can also be written in one-line (but less interpretable):
        # self.state = int(self.state==0)
