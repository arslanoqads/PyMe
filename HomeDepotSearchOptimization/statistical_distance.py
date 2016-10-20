from gensim.models import Word2Vec
model= Word2Vec.load_word2vec_format('/Users/pengdong/Desktop/python/mymodel.bin.gz',binary=True)
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import string
import nltk
#standardize the dataset, including remove the digit,punctuation, switch all to
#lower case, tokenize,the input is a single text
def remove_digit(string0):
    result0 = string0.lower()
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    result0 = result0.translate(replace_punctuation)
    result0 = ''.join([i for i in result0 if not i.isdigit()])
    result0 = nltk.word_tokenize(result0)
    return result0
#get the distance between two text, the return is the average value of
#all the smallest values between query and a target text
def naive_distance(qr,tr):
    if pd.isnull(tr):
        return 1
    query = remove_digit(qr)
    target = remove_digit(tr)
    list_tmp=[]
    for i in range(0,len(query)):
        list_row=[]
        for j in range(0,len(target)):
                try:
                    sim = model.similarity(query[i],target[j])
                except KeyError:
                    sim = 0
                list_row.append(sim)
        list_tmp.append(1-max(list_row))
    naive_distance = np.mean(list_tmp)
    return naive_distance
#get the distance between two features, the output is two columns in the
#Data frame
def get_dist(colq,colt):
    listemp=[]
    for i in range(0,max([len(colq),len(colt)])):
        score = naive_distance(colq[i],colt[i])
        listemp.append(score)
    return listemp

#calculate the jaccard coefficient
def jaccard(a,b):
    A,B=nltk.word_tokenize(a),nltk.word_tokenize(b)
    A,B = set(A),set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = float(intersect)/union
    return coef

#this is to calculate the cos metrics, which is used in the word mover distance
#the input is a text
def cos_metric(features):
    lenth = len(features)
    metrics = np.zeros([lenth,lenth])
    for i in range(0,lenth):
        for j in range(0,lenth):
            try:
                tmp = model.similarity(features[i],features[j])
            except KeyError:
                tmp = 0
            metrics[i][j] = tmp
    return metrics
#This is to standardize the text we will use in the calculation of word mover
#distance,the input is a raw text
def emd_standardize(string0):
    import string
    result0 = string0.lower()
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    result0 = result0.translate(replace_punctuation)
    result0 = ''.join([i for i in result0 if not i.isdigit()])
    return result0

#this is to calculate the word mover distance based on cosine metrics we just calculated
#the input is two text
def emd_cos(d1,d2):
    if pd.isnull(d1) or pd.isnull(d2):
        return 1
    vect = CountVectorizer(stop_words="english").fit([d1, d2])
    v_1, v_2 = vect.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    D = cos_metric(vect.get_feature_names())
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D = D.astype(np.double)
    D /= D.max()
    emd_cosin = emd(v_1,v_2,D)
    return emd_cosin
#This is to calculate the word mover distance based on cosine metrics
#for two columns, input are two columns output is one column
def emd_col(colq,colt):
    listemp=[]
    for i in range(0,max([len(colq),len(colt)])):
        score = emd_cos(colq[i],colt[i])
        listemp.append(score)
    return listemp
#this part is to calculate the word mover distance based on euclidean metrics,
#in this part, sklearn.metrics.euclidean_distance is introduced for calcution

#First, standardize
def emd_standardize(string0):
    result0 = string0.lower()
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    result0 = result0.translate(replace_punctuation)
    result0 = ''.join([i for i in result0 if not i.isdigit()])
    if result0 == None:
        print 'caution! the string is empty!'
    return result0
from sklearn.metrics import euclidean_distances
#calculate the distance between two single texts
def wmd(d1,d2):
    if pd.isnull(d1) or pd.isnull(d2):
        return 1
    d1 = emd_standardize(d1)
    d2 = emd_standardize(d2)
    vect = CountVectorizer(stop_words="english").fit([d1, d2])
    names = vect.get_feature_names()
    v_1, v_2 = vect.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    W_ = []
    for i in range(0,len(names)):
        try:
            W_.append(model[names[i]])
        except KeyError:
            W_.append(np.zeros(300))
    D_ = cosine_distances(W_)
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 = v_1 +1
    v_2 = v_2 +1
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ = D_ +1
    D_ /= D_.max()
    wmd = emd(v_1, v_2, D_)
    return wmd
#Calculate for two columns
def wmd_col(colq,colt):
    listemp=[]
    for i in range(0,max([len(colq),len(colt)])):
        score = wmd(colq[i],colt[i])
        listemp.append(score)
    return listemp
