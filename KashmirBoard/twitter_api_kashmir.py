from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_engine import sentiment
import time




class listener(StreamListener):
    def __init__(self,tag,confidence):
        self.tag=tag
        self.confidence=confidence

    def on_data(self, data):
        all_data=json.loads(data)
        tweet=all_data["text"]
        sent, conf=sentiment(tweet)
        c=0
        if (conf>=90 and sent=='neg') or (sent=='pos'):

            output=open('/twitter_'+self.tag[0]+'.txt','a')
            output.write(sent)
            output.write('\n')
            output.close()
   
#            

        return True

    def on_error(self, status):
        print status


def caller(tag,confidence):    
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)


    twitterStream = Stream(auth, listener(tag,confidence))
    twitterStream.filter(track=tag)

caller(['kashmir'],80)
        
    
    
    
    
    
