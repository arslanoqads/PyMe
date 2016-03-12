import threading

#inherit the threading into your class
class my_class(threading.Thread):
    #this method is default, always runs
    def run(self):
        for _ in range(20):
            print (threading.current_thread().getName())
            
"""
to create multiple threads, create multiple instances of a class, n then initiate
them as threads"""            

x=my_class(name='Arsi\n')
y=my_class(name='ufi\n')

#trigger the threads
x.start()
y.start()           