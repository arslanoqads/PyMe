
#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import pandas as pd
import time



ckey='PIcj4Xn3CSf51IHRKjvUf7mmT'
csecret='ZkJDVSXOARY4qGWqN3w1l0aLR3Zmr3JIrb5Qf2JKzc8AuLZinL'
atoken='27452124-O6svLVkFHRSjgVgwyOQspMomQPQLBWVQJ2cZbPMCA'
asecret='qXh0Y5AZHDPIJU1kByl79Lb0gE7puw6lpLjbavLEjwcy1'
            


class StdOutListener(StreamListener):

    def on_data(self, data):
        text_file = open("/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/kashmir_tweets.txt", "a")
        #text_file = open("log.txt", "a")        
        text_file.write(data)    
        text_file.close()

        
        return True

    def on_error(self, status):
        time.sleep(901)
        stream.filter(track=['kashmir'])

l = StdOutListener()
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
stream = Stream(auth, l)    


stream.filter(track=['kashmir'])




