from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import re


def clean(text):
    
   # text= (unicode(text,'utf-8'))
    temp=text.replace('"',' ')
    temp=temp.replace(',',' , ')
    temp=temp.replace('.',' . ')
    temp=temp.replace('  ',' ')
    temp=temp.replace('<',' ')
    temp=temp.replace('>',' ')
    temp=temp.replace('<',' ')
    temp=temp.replace('”','"')
    temp=temp.replace('“','"')
    text=re.sub( '\s+', ' ', temp).strip().decode('utf-8') #remove multiple spaces
    text=re.sub(r'\\.{1,3}\\.{1,3}\\.{1,4}','',text)
    text=re.sub(r"\\.{6}"," ",text)
    
    return text
    
#    temp=temp.lower()
#    tokens=temp.split(' ')
#    
#    print len(tokens)
#    
#    dict={}
#    for word1 in tokens:
#        dict[word1]=0
#        for word2 in tokens:
#            if word1==word2:
#                dict[word1]=dict[word1]+1
#    
#                
#    word_sent = [word_tokenize(s.lower()) for s in sents]            
        
        
    ##############################################################    
        
 


_min_cut = 0.1
_max_cut = 0.9 
_stopwords = set(stopwords.words('english') + list(punctuation))


def _compute_frequencies(word_sent):

    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in _stopwords:
          freq[word] += 1
       
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in freq.keys():
      freq[w] = freq[w]/m
      if freq[w] >= _max_cut or freq[w] <= _min_cut:
        del freq[w]
    return freq    


def summarize(text, n):
    """
      Return a list of n sentences 
      which represent the summary of text.
    """
    text=clean(text)
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    _freq = _compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in _freq:
          ranking[i] += _freq[w]
    sents_idx = _rank(ranking, n)    
    return [sents[j] for j in sents_idx]
    
def _rank(ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)        

#summarize(text,n)    