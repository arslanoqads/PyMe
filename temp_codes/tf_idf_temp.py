
# http://aimotion.blogspot.com/2011/12/machine-learning-with-python-meeting-tf.html

import random
from nltk.corpus import names
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk 
from nltk.corpus import stopwords
import pandas as pd
import time
import math
from nltk.util import ngrams

from nltk.classify import ClassifierI

allowed_pos=['JJ','JJR','JJS'] 
rev_pos=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\positive.txt","r")
rev_neg=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\negative.txt","r") 

# count of each word in a document
def freq(word, doc):
    return doc.count(word)

# total number of documents
def word_count(doc):
    return len(doc)

# term frequency : normalised count of each word in document
# takes care of issues like multiple similar words in a sentence
def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))

# count of documents containing the word
def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        
        if freq(word, document) > 0:
            count += 1
    return 1 + count

# count the idf : log of number of docs / number of docs containing the word
"""
a high weight of the tf-idf  is reached when you have a 
high term frequency (tf) in the given document and low document
 frequency of the term in the whole collection.
 """
 
def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

# tf-idf
def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))

# read from input opened file
def doc_reader(input):
    doc=[] 
    k=0
    for i in input:
        k+=1
        rows=i.split('\n')
        doc.append(rows)
        if k==15 :
            return doc


# method to find all required metrics in a dictionary format
#  tf, idf,tf_idf
def vocab(all_rev):
    vocab=[]
    
    for rows in all_rev:                  
        words=[word_tokenize(str(i)) for i in rows]  
        words=[i for i in words[0] if i.isalpha() and len(i)>1]
        word=[i for i in words if i not in stopwords.words('english')]
        bigrams=[i for i in ngrams(word,2) if i not in word]
        trigrams=[i for i in ngrams(word,3) if i not in word]         
        word.extend(bigrams)
        word.extend(trigrams)                                  
        vocab.append(word)
    return vocab


def all_metrics(all_rev,vocab):
    dict={}
    c=0
    for rows in all_rev:                  
        c+=1
        words=[word_tokenize(str(i)) for i in rows]  
        words=[i for i in words[0] if i.isalpha() and len(i)>1]
        word=[i for i in words if i not in stopwords.words('english')]
        bigrams=[i for i in ngrams(word,2) if i not in word]
        trigrams=[i for i in ngrams(word,3) if i not in word]         
        word.extend(bigrams)
        word.extend(trigrams)
        dict[str(rows)]=[{
                'word':i,    
                'freq':freq(i,word),
                'tf':tf(i,word),                
                'idf':idf(i,vocab),
                'tf_idf':tf_idf(i,word,vocab),                
                } for i in word]                                         
    return dict   

#dict[c]=[{i:{tf(i,word),idf(i,vocab),tf_idf(i,word,vocab)}} for i in word]

pos=doc_reader(rev_pos)
vocab=vocab(pos)
tfo=all_metrics(pos,vocab)
  
print "Done Collecting"


#bigrams=ngrams(token,2)



  

"""
THINGS READY
1. Set of labeled review/document. all_rev
2. Set of words in positive reviews. pos_words
3. Set of words in negative reviews. neg_words


In the feature method we select only the words that are present 
in the all_words corpus. If none of the words are present we add
a small absilon or usally a one. Eg. in a sentence the probability
of a word, will be 1/n even if the word does not exist. This is 
the solution to the problem of missing values

"""

def feature(review):
    words=word_tokenize(review)
    features={}
    for i in words:
        features[i]=(i in all_words)
    return features    

#find words from a review that are present in the existing set.
#convert reviews into [wordset+category]
#this is a feature
#feature_set=[(feature(str(rev)),cat) for (rev,cat) in all_rev]
#
#
#random.shuffle(feature_set)    
#
#train_set, test_set = feature_set[:6000], feature_set[6000:]


# naive bayes
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print('NB accuracy :',(nltk.classify.accuracy(classifier, test_set))*100)

##MNB
#MNB=SklearnClassifier(MultinomialNB())
#MNB.train(train_set)
#print ('MNB accuracy', (nltk.classify.accuracy(MNB, test_set))*100)
#
#
##BNB
#BNB=SklearnClassifier(BernoulliNB())
#BNB.train(train_set)
#print ('BNB accuracy',(nltk.classify.accuracy(BNB, test_set))*100)
#
#
#
##LR
#LR=SklearnClassifier(LogisticRegression())
#LR.train(train_set)
#print ('LR accuracy',(nltk.classify.accuracy(LR, test_set))*100)
#
#
##SDGC
#SDGC=SklearnClassifier(SGDClassifier())
#SDGC.train(train_set)
#print ('SGDC accuracy',(nltk.classify.accuracy(SDGC, test_set))*100)
#
#
##LSVC
#LSVC=SklearnClassifier(LinearSVC())
#LSVC.train(train_set)
#print ('LSVC accuracy',(nltk.classify.accuracy(LSVC, test_set))*100)
#
#
##SVC
#SVC=SklearnClassifier(SVC())
#SVC.train(train_set)
#print ('SVC accuracy',(nltk.classify.accuracy(SVC, test_set))*100)
#
#
##NSVC
#NSVC=SklearnClassifier(NuSVC())
#NSVC.train(train_set)
#print ('nuSVC accuracy',(nltk.classify.accuracy(NSVC, test_set))*100)
#
#
#
## Saving the objects to pickle:
#with open('classifiers_8_words.pickle', 'wb') as f:
#    pickle.dump([classifier, MNB,BNB,LR,SDGC,LSVC,SVC, NSVC,all_words], f)

# Getting back the objects:

#with open('objs.pickle') as f:
#    classifier, MNB,BNB,LR,SDGC,LSVC,SVC, NSVC = pickle.load(f)



