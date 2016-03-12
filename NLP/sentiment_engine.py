
from nltk.tokenize import word_tokenize
import nltk 
import pickle
import pandas as pd



from nltk.classify import ClassifierI


# voting meachnism
# run data through various algorithms
# called at the end
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifier=classifiers
 #method runs the feature on all algos, checks for the best resuts.       
    def classify (self,features):
        votes=[]
        for c in self._classifier:
            v=c.classify(features)
            votes.append(v)
 #calculates the % of best results           
        mode =max(set(votes), key=votes.count)   
            
        return mode        
        
    def confidence (self,features):
        votes=[]
        for c in self._classifier:
            v=c.classify(features)
            votes.append(v)             
        mode =max(set(votes), key=votes.count)            
        best_choices=votes.count(mode)   
        conf=(best_choices/float(len(votes)))*100
        
        return conf


def feature(review):
    words=word_tokenize(review)
    features={}
    for i in words:
        features[i]=(i in all_words)
    return features    



# Getting back the objects:
with open('classifiers_8_words.pickle') as f:
    classifier, MNB,BNB,LR,SDGC,LSVC,SVC1, NSVC,all_words = pickle.load(f)


def sentiment(review):
    feat=feature(review)
    x=VoteClassifier(classifier,MNB,BNB,LR,SDGC,LSVC,SVC1,NSVC)
#    print 'The classification is: ',x.classify(feature)
#    print 'The confidence is: ',x.confidence(feature)
    return x.classify(feat),x.confidence(feat)
    