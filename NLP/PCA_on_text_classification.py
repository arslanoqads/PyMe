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


from nltk.classify import ClassifierI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA



vectorizer = CountVectorizer(min_df=1)
pca = RandomizedPCA(n_components=30)#, whiten=True)


#collection of all reviews with labels
all_rev=[]   
all_pos=[]
all_neg=[]   
 
allowed_pos=['JJ','JJR','JJS'] 
rev_pos=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\positive.txt","r")
rev_neg=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\negative.txt","r") 
  
for i in rev_pos:    
    try:        
        all_pos.append(unicode(i.strip(),'utf-8'))
    except:
        print 1
    
for j in rev_neg:
    try:
        all_neg.append(unicode(j.strip(),'utf-8'))
    except:
        print 2
    
all_rev=all_pos+all_neg    

def pcs(review):
    X = vectorizer.fit_transform(review)
    pcs = pca.fit_transform(X.toarray())
    print(pca.explained_variance_ratio_) 
    return pcs    

x=pcs(all_rev)

y=[]
for i in x:
    y.append(list(i))
    

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

#MNB
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



