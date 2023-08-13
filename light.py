import random

# State space for individual traffic light = R/G, or in binary 0/1
WAIT_TIME_BEFORE_CHANGE = 10

class Light:
    def __init__(self, time=0):
        self.state = 0 #random.choice([0,1])
        self.time = time
        self.last_changed = 0
        self.wait_time_before_change = WAIT_TIME_BEFORE_CHANGE

    def change_state(self, new_state):
        # The logic can also be written in one-line (but less interpretable):
        # self.state = int(self.state==0)
        self.time += 1
        
        if not self.if_changeable():
            return
        
        # if self.state == 0:
        #     self.state = 1
        # else:
        #     self.state = 0

        self.state = new_state
        
        self.last_changed = self.time
    
    def if_changeable(self):
        # If we want to hardcode to enforce periods when lights are unchangable 
        # ive denoted it as 4 time periods as an example
        if self.time > self.last_changed + self.wait_time_before_change:
            return True
        return False 
    