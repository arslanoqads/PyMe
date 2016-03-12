
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
        print 'mode is ',mode        
        return mode        
        
    def confidence (self,features):
        votes=[]
        for c in self._classifier:
            v=c.classify(features)
            votes.append(v)             
        mode =max(set(votes), key=votes.count)            
        best_choices=votes.count(mode)   
        conf=(best_choices/float(len(votes)))*100
        print votes
        return conf


def feature(review):
    words=word_tokenize(review)
    features={}
    for i in words:
        features[i]=(i in all_words)
    return features    

all_rev=[]   
pos_words=[]
neg_words=[]   
 
rev_pos=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\positive.txt","r")
rev_neg=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\negative.txt","r") 


for i in rev_pos:    
    all_rev.append((i.split('\n'),'pos'))

for i in rev_neg:
    all_rev.append((i.split('\n'),'neg'))

    
all_frame=pd.DataFrame(all_rev, columns=['review','cat'])  

pos_frame=all_frame[all_frame['cat']=='pos']
neg_frame=all_frame[all_frame['cat']=='neg']

for i in pos_frame['review']:
    pos_words.extend(word_tokenize(str(i).lower()))


for i in neg_frame['review']:
    neg_words.extend(word_tokenize(str(i).lower()))       
    
all_words=pos_words+neg_words  



# Getting back the objects:
with open('classifiers_8.pickle') as f:
    classifier, MNB,BNB,LR,SDGC,LSVC,SVC1, NSVC = pickle.load(f)


review=''

def sentiment(review):
    feat=feature(review)
    x=VoteClassifier(classifier,MNB,BNB,LR,SDGC,LSVC,SVC1,NSVC)
#    print 'The classification is: ',x.classify(feature)
#    print 'The confidence is: ',x.confidence(feature)
    return x.classify(feat),x.confidence(feat)
    
sentiment(review)