import random
from nltk.corpus import names
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk 
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import  SVC,LinearSVC,NuSVC
import pandas as pd
import time
import math

from nltk.classify import ClassifierI




#collection of all reviews with labels
all_rev=[]   
pos_words=[]
neg_words=[]   
 
allowed_pos=['JJ','JJR','JJS'] 
rev_pos=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\positive.txt","r")
rev_neg=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\negative.txt","r") 

k=0
for i in rev_pos:
    k+=1    
    all_rev.append((i.split('\n'),'pos'))
    if k==15 : break
k=0
for i in rev_neg:
    k+=1
    all_rev.append((i.split('\n'),'neg'))
    if k==15 : break

    
all_frame=pd.DataFrame(all_rev, columns=['review','cat'])  

pos_frame=all_frame[all_frame['cat']=='pos']
neg_frame=all_frame[all_frame['cat']=='neg']

#pos_frame=pos_frame[:10]
#neg_frame=neg_frame[:10]

for i in pos_frame['review']:
    pos_words.extend(word_tokenize(str(i).lower()))


for i in neg_frame['review']:
    neg_words.extend(word_tokenize(str(i).lower()))       

neg_words=[i for i in neg_words if i.isalpha()]
pos_words=[i for i in pos_words if i.isalpha()] 
   
all_words=pos_words+neg_words    
print "Done Collecting"


def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))










  

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



