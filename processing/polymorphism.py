#! /usr/bin/python


"""
The code snippet shows the concept of inheritence and how some methods can be
over ridden in the subclasses, at the same time it shows how methods from the 
super class can be called if the name of the methods is the same in both of 
them. 

It demonstrates polymorphism by passing objects in a method and the calling 
some methods inside the method for different classes.

It also demonstrates how the accessors or the getter and setter function can
be used. 

-__name cannot be accessed by the class object directly
-__metaclass = type is necessary for the supe() functionality to work in non
3.0 Python

"""

__metaclass__ = type

class Animal:
    __name = "No Name"        # the code will automatically declare even if this is commented
    __owner = "No Owner"      # the code will automatically declare even if this is commented
    meat='nome'


    # it takes multiple dictionary values
    def __init__(self, **kvargs): # The constructor function called when object is created
         self._attributes = kvargs  #unpacks the dictionary values into the _attributes
         print 'Animal init, that one'
    
    # There is a function called a destructor __del__, but its best to avoid it
    # The next two methods automatically declare and set variables. The variable 
    # names need not have been declared.
    def set_attributes(self, key, value): # Accessor Method
         self._attributes[key] = value
         return
    
    def get_attributes(self, key):
         return self._attributes.get(key, None)
    
    def noise(self): # self is a reference to the object
         print('errr') # You use self so you can access attributes of the object
         return
    
    def move(self):
         print('The animal moves forward')
         return
    
    def eat(self):
         print('Crunch, crunch')
         return
    
    def __hiddenMethod(self): # A hidden method
         print "Hard to Find"
         return

class Dog(Animal):

    def __init__(self, **kvargs):   # Not needed unless you plan to override the super
         super(Dog, self).__init__() # This wouldn't work in 2.7 without the second _meta..
         self._attributes = kvargs
         print 'Dog init that one'
    
    def noise(self):        # Overriding the Animal noise function
         print('Woof, woof')
         Animal.noise(self)
         return

class Cat(Animal):

    def __init__(self, **kvargs):  # Not needed unless you plan to override the super
         #super(Cat, self).__init__()
         self._attributes = kvargs
         print 'Cat init that one'
    
    def noise(self):
         print('Meow')
         return
    
    def noise2(self):
         print('Purrrrr')
         return

class Dat(Cat,Dog):

    def __init__(self, **kvargs):  # Not needed unless you plan to override the super
         super(Dat, self).__init__()
         self._attributes = kvargs
    
    def move(self):
         print('Chases Tail')
         return
    
    def playWithAnimal(Animal): # This is polymorphism
         Animal.noise()
         Animal.eat() # Works even if the method isn't in Cat because Cat is an Animal
         Animal.move()
         print(Animal.get_attributes('__name'))
         print(Animal.get_attributes('__owner'))
         print '\n'
         Animal.set_attributes('clean',"Yes")
         print(Animal.get_attributes('clean'))

    print 'check 1' # first animal and theg dog constructor is called.
    jake = Dog(__name = 'Jake', __owner = 'Paul')
    
    print 'check 2'  #call to animal constructor is commented, so only cat is called.  
    sophie = Cat(__name = 'Sophie', __owner = 'Sue')
    
    print 'check 3'    
    playWithAnimal(sophie)
    playWithAnimal(jake)
    
    # print sophie.__hiddenMethod() Demonstrating private methods
    
    print issubclass(Cat, Animal) # Checks if Cat is a subclass of Animal
    print Cat.__bases__           # Prints out the base class of a class
    print sophie.__class__        # Prints the objects class
    print sophie.__dict__         # Prints all of an objects attributes
    
japhie = Dat(__name = 'Japhie', __owner = 'Sue')
japhie.move()
print japhie.get_attributes('__name')
