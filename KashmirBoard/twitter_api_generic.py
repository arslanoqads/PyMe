from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_engine import sentiment
import time

ckey='LX3JE2zaQkb3nqmxeBMc8x09B'
csecret='7W3YI5o0kSeZUohS7yZCdSPlkyoXzeTIeEE3pto7rQ47nSqfSp'
atoken='746068683110420480-gREhrFUUgQyAR7PYRuQ4jJVL9W7Nhv4'
asecret='WxD0dzl5VsrfSm381AM2C4e9qQ4ppWuIUlJUMrGpJS3xz'


class listener(StreamListener):
    def __init__(self,tag,confidence):
        self.tag=tag
        self.confidence=confidence

    def on_data(self, data):
        all_data=json.loads(data)
        tweet=all_data["text"]
        sent, conf=sentiment(tweet)
        
        if (conf>=85 and sent=='neg') or (sent=='pos'):
            try:
                output=open('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/twitter_'+self.tag[0]+'.txt','a')
                output.write(sent)
                output.write('\n')
                output.close()
                print 'posted for '+self.tag[0]
         
                return False       
            except Exception as e:
                output=open('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/error_'+self.tag[0]+'.txt','a')
                output.write(e)
                output.write('\n')
                output.close() 
        return True

    def on_error(self, status):
        print status


def caller(tag,confidence):    
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)


    twitterStream = Stream(auth, listener(tag,confidence))
    twitterStream.filter(track=tag)


for i in range(13):

    c=[]
    caller(['tesla'],70)
    
    c=[]
    caller(['datascience'],70)        
    
    c=[]
    caller(['donaldtrump'],70)    
    
    c=[]
    caller(['kashmir'],70)    
    
    c=[]
    caller(['google'],70)        
    
    c=[]
    caller(['facebook'],70)    
    
    c=[]
    caller(['aliens'],70)  
           
    
 
    