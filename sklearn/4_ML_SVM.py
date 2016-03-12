import sklearn
import numpy as np
from sklearn import svm
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.svm import  SVC
from nltk.classify.scikitlearn import SklearnClassifier

style.use('ggplot')



w=np.array([[1,2],
[2,1],
[3,4],
[8,11],
[9,14],
[10,12],
[1,4],
[7,6],
[5,3],
[2,3],
[6,5]
])


z=[1,1,1,0,0,0,1,0,1,1,0]

clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(w,z)

print 'clf',clf.predict([21,20])


"""
maths

c=clf.coef_[0]      #coefficients
a=-c[0]/c[1]        #learning rate
int=clf.intercept_[0]

X=np.linspace(1,12) #generates linearly spaced numbers, just like range()

Y=a*X -int/c[1]


plt.plot(X,Y,'k-',label='dhamaka')
plt.scatter(w[:,0],w[:,1])
plt.show()
"""

#plt.scatter(x,y)

#plt.show()