import twitter_api
import threading
import time

"""
This script calls twitter api load load tweets into files, in multiple threads.
"""


# need this package : pip install pyopenssl ndg-httpsclient pyasn1
hashtags=[['datascience'],['Kashmir'],['Tesla'],['DonaldTrump']]


class getdata1(threading.Thread):  
    def run(self):
        try:            
            twitter_api.caller(hashtags[0],85)            
        except:
            time.sleep(2)
            

class getdata2(threading.Thread):  
    def run(self):
        try:
            
            twitter_api.caller(hashtags[1],85)
            
        except:
            time.sleep(2)
            


class getdata3(threading.Thread):  
    def run(self):
        try:
            
            twitter_api.caller(hashtags[2],85)
            
        except:
            time.sleep(2)
            


class getdata4(threading.Thread):  
    def run(self):
        try:
            
            twitter_api.caller(hashtags[3],85)
            
        except:
            time.sleep(2)


def caller():  
    x=getdata1()
    y=getdata2()
    z=getdata3()
    w=getdata4()
    
    
    
    #trigger the threads
    x.start()
    y.start()
    z.start()
    w.start()
#plt.legend()
