
#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import pandas as pd
import time




            


class StdOutListener(StreamListener):

    def on_data(self, data):
        text_file = open("kashmir_tweets.txt", "a")
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




