#read json
import json
import pandas as pd
import sys


tweets_data = []

tweets_file=open('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/'+str(sys.argv[1]),"r")

#tweets_file = open('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/kashmir_tweets.txt', "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

tweets_file.close() 
  
#remove dirty tweets  
d=[]  
for i in tweets_data:
    if len(i)>10:
        d.append(i)
        
tweets_data=d  

#extract tweet fields
tweets = pd.DataFrame()
tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['verified'] = map(lambda tweet: tweet['user']['verified'], tweets_data)   
tweets['screen_name'] = map(lambda tweet: tweet['user']['screen_name'], tweets_data) 
tweets['created_at'] = map(lambda tweet: tweet['created_at'], tweets_data) 
tweets['entities'] = map(lambda tweet: tweet['entities'], tweets_data) 
tweets['name'] = map(lambda tweet: tweet['user']['name'], tweets_data) 
tweets['followers_count'] = map(lambda tweet: tweet['user']['followers_count'], tweets_data) 
tweets['location'] = map(lambda tweet: tweet['user']['location'], tweets_data) 
tweets['time_zone'] = map(lambda tweet: tweet['user']['time_zone'], tweets_data) 
tweets['friends_count'] = map(lambda tweet: tweet['user']['friends_count'], tweets_data) 
tweets['utc_offset'] = map(lambda tweet: tweet['user']['utc_offset'], tweets_data) 



tweets.to_csv('/var/lib/openshift/56bc1b8c0c1e667ea0000027/app-root/data/kashmir_tweets_pd.csv',encoding='utf-8',header=False,mode='a')



"""

#cleaning csv

l=[]
with open('sep18.csv', 'rU')as csvfile:
    spamreader = csv.reader(csvfile,dialect=csv.excel_tab, delimiter=',', quotechar='"')
    for row in spamreader:
        l.append(row)
        
        

#analysis

#x.columns=['null','text','verified','screen_name','created_at','entities','name','followers_count','location','time_zone','friends_count','utc_offset']

y=x[x.verified.isin(['True','TRUE','FALSE','False'])]


y.created_at=y.created_at.apply(lambda x : x if str(x)[-3:-1]=='01' else '')
y.created_at=pd.to_datetime(y.created_at,infer_datetime_format=True)
 
y.created_at=y.created_at+pd.Timedelta(hours=-5.5)
 
y.created_at=y.created_at.apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
 
 
"""


 