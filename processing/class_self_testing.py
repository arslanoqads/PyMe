


class my_class():
    
    def __init__(self):
        print 'chasti aaraha hai'
        
    def add(self,a,b,c):
        self.a=a
        self.b=b #these variables can be accessed as object1.b to give the value outside the class
        print 'lelo',self.ad(4,5) # need self to access ad
       # print 'koko', ad(4,5)  #ad method cannot be accessed here without self
        #to access any method/vairable inside the class we need to access using self.
        return self.a + self.b 
        
        
    def ad(self, a1,b1):
        return a1+b1
        
class his_class(my_class):
    def __init__(self):
        print 'lunky aaraha hai'
    
    pass