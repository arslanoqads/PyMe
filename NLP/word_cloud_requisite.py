import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk import pos_tag

import re

file= open('jobs.txt','r')

f1=[]

for line in file:
    line=line.strip()
    line=filter(None,re.split('[],.-/\\_ ]+',line))  #splitting based on multiple delimiters
    #line=line.split(' ')
    for i in line:
        f1.append(i.lower())
        


 
all_words=[]

#
#print 'all words before any cleaning:',len(f1)
swords=stopwords.words()
all_words=[i for i in f1 if i.isalpha()]    

#print 'total words after removing punctuations:',len(all_words)
all_words=[i for i in all_words if i not in swords]  



#print 'before:',len(all_words)

#pick up only nouns
all_nouns=[w for w,t in pos_tag(all_words) if t in ['NN','NNS','NNP','NNPS']]

final_set=' '.join(all_nouns)
#print 'nouns',len(all_nouns)
#
#all=nltk.FreqDist(all_nouns) 
#
#
#
#
#counts=pd.DataFrame(all.values())
#words=pd.DataFrame(all.keys())
#
#freqs=pd.concat([words,counts],axis=1, join_axes=[words.index])         
#freqs.columns=['words','counts']
#freqs.sort(columns='counts',inplace=True)
