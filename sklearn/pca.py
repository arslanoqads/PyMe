"""
for numbers
"""

#import numpy as np
#from sklearn.decomposition import PCA
##array=[[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
#array=[['are','they'],['may','you']]
#X = np.array(array)
#pca = PCA(n_components=2)
#pca.fit(X)
##PCA(copy=True, n_components=2, whiten=False)
#print(pca.explained_variance_ratio_) 
##[ 0.99244...  0.00755...]


"""
other
"""


sentences = [
    "fix grammatical or spelling errors",
    "clarify meaning without changing it",
    "correct minor mistakes",
    "add related resources or links",
    "always respect the original author"
]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA

# Initialize models
vectorizer = CountVectorizer(min_df=1)
pca = RandomizedPCA(n_components=5)#, whiten=True)

km = KMeans(n_clusters=2, init='random', n_init=1, verbose=1)

# Fit models
X = vectorizer.fit_transform(sentences)
#X2 = pca.fit_transform(X.toarray())
X2 = pca.fit(X.toarray())
#km.fit(X2)
print(pca.explained_variance_ratio_) 

# Predict with models
#X_new = vectorizer.transform(["hello world"])
#X2_new = pca.transform(X_new)
#km.predict(X2_new)




