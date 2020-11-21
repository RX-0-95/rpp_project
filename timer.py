import time 
#This class is a singletion class
#timer: calcualte the time elasepd between each 2 call 
#In this project is used to calculte the fps of the video 
#or camear, and the fps of captured face frame 
class timer():
    __instance = None 
    @staticmethod 
    def timerInstance():
        if timer.__instance == None:
            timer() 
        return timer.__instance
    
    def __init__(self):
        if timer.__instance != None:
            raise Exception("Timer class is a singleton class")
        else: 
            timer.__instance = self 
            self.prev_time = 0 
            self.last_time = 0
            self.next_time = 0
            self.frame_time = 0
            self.time_elapsed = 0 
            self.start_time = 0 
            self.last_time_in_time_elapsed = 0 
            self.current_time = 0 
            self.started = False 

    def start(self):
        self.started = True 
        self.time_elapsed = 0
        self.last_time  = time.time() 
        self.last_time_in_time_elapsed = self.last_time 
        self.start_time  = self.last_time 
        self.current_time = self.last_time 

    def timeElapsed(self):
        self.current_time = time.time()
        self.time_elapsed = self.current_time - self.last_time_in_time_elapsed 
        self.last_time_in_time_elapsed = self.current_time 
        return self.time_elapsed


