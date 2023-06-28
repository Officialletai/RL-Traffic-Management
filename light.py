# State space for individual traffic light = R/G, or in binary 0/1

class Light:
    def __init__(self):
        self.state = 0
        self.last_changed = 0

    def change_state(self, current_time):
        # The logic can also be written in one-line (but less interpretable):
        # self.state = int(self.state==0)
        if self.state == 0:
            self.state = 1
        else:
            self.state = 0
        
        self.last_changed = current_time
    
    def if_changable(self, current_time):
        # If we want to hardcode to enforce periods when lights are unchangable 
        # ive denoted it as 4 time periods as an example
        if current_time > self.last_changed + 4:
            return True
        return False 
    

        

