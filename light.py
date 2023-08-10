# State space for individual traffic light = R/G, or in binary 0/1

class Light:
    def __init__(self):
        self.state = 0 #random.choice([0,1])
        self.last_changed = 0
        self.time = 0

    def change_state(self):
        # The logic can also be written in one-line (but less interpretable):
        # self.state = int(self.state==0)
        self.time += 1
        
        if not self.if_changable():
            return
        
        if self.state == 0:
            self.state = 1
        else:
            self.state = 0
        
        self.last_changed = self.time
    
    def if_changable(self):
        # If we want to hardcode to enforce periods when lights are unchangable 
        # ive denoted it as 4 time periods as an example
        if self.time > self.last_changed + 10:
            return True
        return False 
    

        

