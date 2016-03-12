from sklearn import svm, datasets
import matplotlib.pyplot as plt



"SVM"
digits=datasets.load_digits()
clf=svm.SVC(gamma=0.0001,C=1000000)
x,y=digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
print ('Predict :',clf.predict(digits.data[-1]))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r)
plt.show()


