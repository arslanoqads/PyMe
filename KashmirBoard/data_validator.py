
import re
import pandas as pd
import ast
import sys


# this file takes the input from the output of the tweet_counter file, extracts
# mentions and tags from entities.

file_in=sys.argv[1] 

q=pd.read_csv(file_in)


q.columns=['text','verified','screen_name','created_at','entities','name','followers_count','location','time_zone','friends_count','utc_offset']

q.drop_duplicates(inplace=True)
q.reset_index(inplace=True)
del q['index']


#too keep the data clean, while loading all commas have been replaced by dots
q.entities=q.entities.apply(lambda x: str(x).replace('. ',' ,'))

# add columns
q['mentions']=''
q['tags']=''
q['json']=''

#clean data
try:
q.text=q.text.apply(lambda x : str(x))
q.text=q.text.apply(lambda x : x.replace('RT @','@'))
q.text=q.text.apply(lambda x : re.sub('@[\S]*','',x))
q.text=q.text.apply(lambda x : re.sub('[\S]*#[\S]*','',x))
q.text=q.text.apply(lambda x : re.sub('https[\S]*','',x))
q.text=q.text.apply(lambda x : x.encode('string-escape')) #convert to raw : for unicode/ascii problem
q.text=q.text.apply(lambda x : re.sub(r"\\[a-z]{1,2}\d{1,2}",'', x))
q.text=q.text.apply(lambda x : x.replace('\\',''))
q.text=q.text.apply(lambda x : x.replace('  ',''))
q.text=q.text.apply(lambda x : x.lower())

except :
    pass
q=q[q.verified.isin(['True','TRUE','FALSE','False',True,False])]

def f(x):
    try : 
        return ast.literal_eval(x) 
    except :
        return 1

q.json=q.entities.apply(f)

clean=q[q.json!=1]
dirty=q[q.json==1]

#file specific as some jsons dont convert with valid values
clean['len']=clean.json.apply(lambda x : len(str(x)))
clean.sort('len',inplace=True)
clean=clean[clean.len>16]

# get a list of mentions and tags
clean.mentions=clean.json.apply(lambda x :[i['screen_name'] for i in x['user_mentions']])

clean.tags=clean.json.apply(lambda x :[i['text'] for i in x['hashtags']])

clean=clean[['text','verified','screen_name','created_at','name','followers_count','location','time_zone','friends_count','utc_offset','mentions','tags']]


clean.created_at=pd.to_datetime(clean.created_at,infer_datetime_format=True)


clean.to_csv('date_fixed.csv',mode='a')
