from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_engine import sentiment

ckey='xx'
csecret='xx'
atoken='xx-xx'
asecret='xx'


class listener(StreamListener):
    def __init__(self,tag):
        self.tag=tag

    def on_data(self, data):
        all_data=json.loads(data)
        tweet=all_data["text"]
        sent, conf=sentiment(tweet)
        if conf>=75:
            output=open('sentiments\\twitter_'+self.tag[0]+'.txt','a')
            output.write(sent)
            output.write('\n')
            output.close()
            
        print sent, conf, self.tag[0]
        return True

    def on_error(self, status):
        print status


def caller(tag):    
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)


    twitterStream = Stream(auth, listener(tag))
    twitterStream.filter(track=tag)
    
    