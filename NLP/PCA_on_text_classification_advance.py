import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import accuracy_score
from sklearn import svm



#################################################
# Cleaning and collecting data

#collection of all reviews with labels
all_rev=[]   
all_pos=[]
all_neg=[]   
 

# POsitive Reviews
rev_pos=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\positive.txt","r")

# Negative Reviews
rev_neg=open("C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\short_reviews\\negative.txt","r") 

# Cleaning Data
po=0  
for i in rev_pos: 
    
    try:
        all_pos.append(unicode(i.strip(),'utf-8'))        
        
    except:
        po+=1
            
    #if k==15:break    
ne=0    
for j in rev_neg:
    
    try:
        all_neg.append(unicode(j.strip(),'utf-8'))     
    except:
        ne+=1
            
print 'missed - ',ne
print 'missed + ',po            
    

#collect all negative and positive reviews        
all_rev=all_pos+all_neg   
X=[] 
print 'check 0'


#################################################
# Initiate Principal Components

vectorizer = CountVectorizer(min_df=1)
pca = RandomizedPCA(n_components=130, whiten=True)

# Method to create components
def pcs(review):
    X = vectorizer.fit_transform(review)
    pcs = pca.fit_transform(X.toarray())
    return pcs    

x=pcs(all_rev)

# Convert Numpy Array to list
# append targets to features
y=[]
for i in x[:len(all_pos)]:        
    y.append(list(i))
    
for i in x[len(all_pos):]:        
    y.append(list(i))
    

# Create target values
p=['p']*len(all_pos)
n=['n']*len(all_neg)


# join features and target value
tot=p+n
 
 
# Fit SVM 
clf = svm.SVC(kernel='rbf',C=5., gamma=0.001)
clf.fit(y[:9000],tot[:9000])

clf.score(y[9000:],tot[9000:])



