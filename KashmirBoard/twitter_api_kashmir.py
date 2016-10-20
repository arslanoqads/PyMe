from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_engine import sentiment
import time

ckey='TsupdEVWLpx3kwYEJf14DkI69'
csecret='i250OKmgha75ymqNEuacSHmC4EFZGLuBpijp86VSeYbCcstG8u'
atoken='2371765364-WkYDXwTrGkorzWFWwvgCaVQ0iqtqAhraaBWLiJ5'
asecret='ibnmdeFRCUiQUmbGUrIkjaigJ7FYmw8dwYoXb7I9ahgoy'


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

            output=open('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/twitter_'+self.tag[0]+'.txt','a')
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
        
    
    
    
    
    