from sklearn.feature_extraction.text import TfidfVectorizer as tv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from nltk.corpus import stopwords, movie_reviews
import nltk
import random
from collections import Counter


text='ab b c c'

cat_prob=0.5

#1 . find probability of each word in a category
list={}
def probalizer(text,category):
    
    word=word_tokenize(text)
    length=len(word)
    count=Counter(word)
    for key, value in count.items():
        print 'key :',key
        print 'value :',value
        print 'len :',length
        list[key]=float(value)/length
        
probalizer(text,1)

#2. for a test sentence find probability of each word, and multiply 
# it by the category probability




# using group by from itertools
x=1
for key,value in list.items():
    x*=value    

print x*cat_prob

from itertools import groupby

x={'a':1, 'b':3,'a':3,'b':4}

for i in groupby(x):
    for j in j:
        
    print i.


