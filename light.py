# State space for individual traffic light = R/G, or in binary 0/1

class Light:
    def __init__(self, current_time):
        self.state = 0
        self.last_changed = 0
        self.current_time = current_time

    def change_state(self):
        # The logic can also be written in one-line (but less interpretable):
        # self.state = int(self.state==0)
        if self.state == 0:
            self.state = 1
        else:
            self.state = 0
        
        self.last_changed = self.current_time
    
    def if_changable(self):
        # If we want to hardcode to enforce periods when lights are unchangable 
        # ive denoted it as 4 time periods as an example
        if self.current_time > self.last_changed + 4:
            return True
        return False 
    

        

