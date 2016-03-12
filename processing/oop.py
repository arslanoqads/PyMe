class program():

    def __init__(self, *args,**kwargs):
        self.name=raw_input("What is your name ?")
        self.country=raw_input("What is ur country ?")
        
        
        
    def prn(self):
        x="27"
        print("Age is ", x)
        print('Name is ',self.name)
        print('From ',self.country)
        
p1=program()        