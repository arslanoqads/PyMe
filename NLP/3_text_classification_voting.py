import random
from nltk.corpus import names
import nltk 
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import  SVC,LinearSVC,NuSVC


from nltk.classify import ClassifierI


# voting meachnism
# run data through various algorithms
#called at the end
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
        
        

#fucntion one to create features from words
def gender_features(word):
    return {'last_letter': word[-1]}


#fucntion two to create features from words    
def gender_features1(word):
    return {'last_2': word[-2:], 'first_2':word[:2]}

#create a labeled list    
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])

#shuffler names
random.shuffle(labeled_names)    

#create feature set
featuresets = [(gender_features1(n), gender) for (n, gender) in labeled_names]

#create test and training sets
train_set, test_set = featuresets[:4000], featuresets[3500:]


"""
for multiple classifiers create a training instance 'classifier' as below.

"""

#train and compare various algorithms
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('NB accuracy :',(nltk.classify.accuracy(classifier, test_set))*100)


MNB=SklearnClassifier(MultinomialNB())
MNB.train(train_set)
print ('MNB accuracy', (nltk.classify.accuracy(MNB, test_set))*100)



BNB=SklearnClassifier(BernoulliNB())
BNB.train(train_set)
print ('BNB accuracy',(nltk.classify.accuracy(BNB, test_set))*100)

#from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklean.svm import  SVC,LinearSVC,NuSVC

LR=SklearnClassifier(LogisticRegression())
LR.train(train_set)
print ('LR accuracy',(nltk.classify.accuracy(LR, test_set))*100)


SDGC=SklearnClassifier(SGDClassifier())
SDGC.train(train_set)
print ('SGDC accuracy',(nltk.classify.accuracy(SDGC, test_set))*100)


LSVC=SklearnClassifier(LinearSVC())
LSVC.train(train_set)
print ('LSVC accuracy',(nltk.classify.accuracy(LSVC, test_set))*100)


SVC=SklearnClassifier(SVC())
SVC.train(train_set)
print ('SVC accuracy',(nltk.classify.accuracy(SVC, test_set))*100)



NSVC=SklearnClassifier(NuSVC())
NSVC.train(train_set)
print ('nuSVC accuracy',(nltk.classify.accuracy(NSVC, test_set))*100)




#save the classifier into a pickle object
save_classifier=open('nb_name_classifier.pickle','wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

#test a prediction
predict=classifier.classify(gender_features('abdullah'))



#open pickle
classifier_opened=open('nb_name_classifier.pickle','rb')
classf=pickle.load(classifier_opened)
classifier_opened.close()

print classf.classify(gender_features1('arslan'))
print 'here'


feature=gender_features1('abdullah')
x=VoteClassifier(classifier,MNB,BNB,LR,SDGC,LSVC,SVC,NSVC)
print 'The classification is: ',x.classify(feature)
print 'The confidence is: ',x.confidence(feature)


