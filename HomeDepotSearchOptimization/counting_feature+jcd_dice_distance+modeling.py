from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.collocations import *
import nltk, string
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
import sklearn.metrics as metrics
from nltk.metrics.distance import jaccard_distance,edit_distance
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize# import data
import time
cachedStopWords = stopwords.words("english")
def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for
    each variable is null and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return (pd.crosstab(df_lng.variable, null_variables))
%matplotlib inline
import matplotlib.pyplot as plt


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


def try_divide(x, y, val=0.0):
    """
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def JaccardCoef(A, B):
    A, B = set(nltk.word_tokenize(A)), set(nltk.word_tokenize(B))
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(nltk.word_tokenize(A)), set(nltk.word_tokenize(B))
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d


# change search term 'e' to 'weber'
df_all.loc[df_all['search_term_spell_checked'] == 'e','search_term_spell_checked'] = 'weber'

df_all['product_info'] = df_all['search_term_spell_checked']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']+'\t'+df_all['product_attributes']
df_all['attr'] = df_all['search_term_spell_checked']+"\t"+df_all['brand']+"\t"+df_all['material']+"\t"+df_all['color']

#counting features
df_all['len_of_query'] = df_all['search_term_spell_checked'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_material'] = df_all['material'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_color'] = df_all['color'].map(lambda x:len(x.split())).astype(np.int64)


#counting features continued,query stands for whole query as a string, word stands for single word in query
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['query_in_attributes'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[3],0))
df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
df_all['query_last_word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[3]))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))

df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['ratio_attributes'] = df_all['word_in_attributes']/df_all['len_of_query']

#tackle brand, material, color
df_all['query_in_brand'] = df_all['attr'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']

df_all['word_in_material'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_material'] = df_all['word_in_material']/df_all['len_of_material']

df_all['word_in_color'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
df_all['ratio_color'] = df_all['word_in_color']/df_all['len_of_color']


# adding bigram, trigram counting features,not including attributes
df_all['search_term_spell_checked_bigram'] = df_all['search_term_spell_checked'].map(lambda x:bigram(x))
df_all['product_title_bigram'] = df_all['product_title'].map(lambda x:bigram(x))
df_all['product_description_bigram'] = df_all['product_description'].map(lambda x:bigram(x))


df_all['len_of_query_bigram'] = df_all['search_term_spell_checked_bigram'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info_bigram'] = df_all['search_term_spell_checked_bigram']+"\t"+df_all['product_title_bigram'] +"\t"+df_all['product_description_bigram']
df_all['query_in_title_bigram'] = df_all['product_info_bigram'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description_bigram'] = df_all['product_info_bigram'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title_bigram'] = df_all['product_info_bigram'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description_bigram'] = df_all['product_info_bigram'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title_bigram'] = df_all['word_in_title']/df_all['len_of_query_bigram']
df_all['ratio_description_bigram'] = df_all['word_in_description']/df_all['len_of_query_bigram']

df_all['brand_bigram'] = df_all['brand'].map(lambda x:bigram(x))
df_all['material_bigram'] = df_all['material'].map(lambda x:bigram(x))
df_all['color_bigram'] = df_all['color'].map(lambda x:bigram(x))
df_all['attr_bigram'] = df_all['search_term_spell_checked']+"\t"+df_all['brand_bigram']+"\t"+df_all['material_bigram']+"\t"+df_all['color_bigram']

df_all['query_title_jcd_unigram'] = df_all['product_info'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[1]))

df_all['query_description_jcd_unigram'] = df_all['product_info'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[2]))
df_all['query_brand_jcd_unigram'] = df_all['attr'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[1]))
df_all['query_material_jcd_unigram'] = df_all['attr'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[2]))

df_all['query_color_jcd_unigram'] = df_all['attr'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[3]))
df_all['query_title_jcd_bigram'] = df_all['product_info_bigram'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[1]))
df_all['query_description_jcd_bigram'] = df_all['product_info_bigram'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[2]))
df_all['query_brand_jcd_bigram'] = df_all['attr_bigram'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[1]))
df_all['query_material_jcd_bigram'] = df_all['attr_bigram'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[2]))
df_all['query_color_jcd_bigram'] = df_all['attr_bigram'].map(lambda x:JaccardCoef(x.split('\t')[0],x.split('\t')[3]))

df_all['query_title_dice_unigram'] = df_all['product_info'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[1]))
df_all['query_description_dice_unigram'] = df_all['product_info'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[2]))
df_all['query_brand_dice_unigram'] = df_all['attr'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[1]))
df_all['query_material_dice_unigram'] = df_all['attr'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[2]))
df_all['query_color_dice_unigram'] = df_all['attr'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[3]))
df_all['query_title_dice_bigram'] = df_all['product_info_bigram'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[1]))
df_all['query_description_dice_bigram'] = df_all['product_info_bigram'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[2]))
df_all['query_brand_dice_bigram'] = df_all['attr_bigram'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[1]))
df_all['query_material_dice_bigram'] = df_all['attr_bigram'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[2]))
df_all['query_color_dice_bigram'] = df_all['attr_bigram'].map(lambda x:DiceDist(x.split('\t')[0],x.split('\t')[3]))


# Modeling

kf = KFold(n = x_train.shape[0] , n_folds=10)
linear_rg = linear_model.LinearRegression(normalize = True)
#linear_rg_bag = BaggingRegressor(linear_rg, n_estimators=100, max_samples=0.1, random_state=25)

scores = cross_val_score(linear_rg,x_train,y_train,scoring = RMSE,cv=10)
#for train, test in kf:
    #rfr_bag.fit(x_train[train],y_train[train])
    #scores = cross_val_score(rfr_bag,x_train[test],y_train[test],scoring = RMSE)

print (scores)
print (scores.mean())

from sklearn.linear_model import Ridge
ridge = Ridge(normalize=False)
scores = cross_val_score(ridge,x_train,y_train,scoring = RMSE,cv=10)
print (scores)
print (scores.mean())

#bayes ridge
from sklearn.linear_model import BayesianRidge
bayes_ridge = BayesianRidge()
bayes_ridge_scores = cross_val_score(bayes_ridge,x_train,y_train,scoring = RMSE,cv=10)
print (bayes_ridge_scores)
print (bayes_ridge_scores.mean())

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_scores = cross_val_score(dt,x_train,y_train,scoring = RMSE,cv=10)
print (dt_scores)
print (dt_scores.mean())

from sklearn.svm import SVR
svr = SVR(C=1.0, epsilon=0.2)
scores = cross_val_score(svr,x_train,y_train,scoring = RMSE,cv=2)

print (scores)

#boosting with bagging

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=3,n_estimators=100,learning_rate = 0.01)
#gbr_bag = BaggingRegressor(gbr, n_estimators=45, max_samples=0.1, random_state=25,n_jobs=-1)
gbr_scores = cross_val_score(gbr,x_train,y_train,scoring = RMSE,cv=2)
print (gbr_scores)
print (gbr_scores.mean())

#for fitting curve, holdout 10%, 0.4766, 700+6+0.01
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
offset = int(x_train.shape[0] * 0.9)
x_train_fit, y_train_fit = x_train[:offset], y_train[:offset]
x_test_fit, y_test_fit = x_train[offset:], y_train[offset:]

clf = GradientBoostingRegressor(n_estimators = 700,max_depth=6,learning_rate = 0.01,subsample=1.0 )

clf.fit(x_train_fit, y_train_fit)
rmse = fmean_squared_error(y_test_fit,clf.predict(x_test_fit))

print("RMSE: %.4f" % rmse)

#ploting， learning rate 0.01 see which n_estimatores should choose, then grid search the max_depth, then do bagging on that
test_score = np.zeros((700,), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test_fit)):
    test_score[i] = clf.loss_(y_test_fit, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(700) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(700) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()

from sklearn import pipeline, grid_search
import time

start_time = time.time()
param_grid = {'max_depth': [6,8,10,12],'n_estimators': [400,500,60]}
clf = GradientBoostingRegressor(n_estimators = 480,max_depth=5,learning_rate = 0.01,subsample=1.0 )
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
model.fit(x_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(x_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_gbr_on_103_grid_search.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))

#rfr.feature_importances_

feat_impt = rfr.feature_importances_

df_feat = df_train.drop(['id','relevance'],axis=1)
feat_names = list(df_feat.columns)
for i in range(len(feat_names)):
    feat = feat_names[i]
    impt = feat_impt[i]
    print (feat + ' ' + str(impt))
std = np.std([rfr.feature_importances_ for tree in rfr.estimators_],
             axis=0)
indices = np.argsort(feat_impt)[::-1]

# Print the feature ranking
print("Feature ranking:")

feat_names_80 = []
cum_impt = 0.0
for f in range(len(feat_names)):
    cum_impt += feat_impt[indices[f]]
    if cum_impt <= 0.8:
        feat_names_80.append(feat_names[indices[f]])
    print("%d. feature %d %s (%f) (%f)" % (f + 1, indices[f], feat_names[indices[f]], feat_impt[indices[f]],cum_impt))

%matplotlib inline
import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(feat_names)), feat_impt[indices],
       color="r", align="center")
plt.xticks(range(len(feat_names)), indices)
plt.xlim([-1, len(feat_names)])
plt.savefig('feat_impt.png')
plt.show()

from sklearn import pipeline, grid_search
import time
start_time = time.time()
param_grid = {'max_features': [8,10,12], 'max_depth': [10,15,20,25]}
rfr = RandomForestRegressor(n_estimators=1000, random_state=0,max_features = 10, max_depth = 14, n_jobs=-1)
model = grid_search.GridSearchCV(estimator = rfr, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
model.fit(x_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(x_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_rfr_on_105_90%_grid_search.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))


#0.8 impt feat, rfr n_estimators=1000, random_state=0,max_features = 10, max_depth = 20, n_jobs=-1
#feat_names_80.append('id')
#feat_names_80.append('relevance')
df_all_test_80 = df_all_test[feat_names_80]
df_train = df_all_test_80.iloc[:train_row]
df_test = df_all_test_80.iloc[train_row:]
id_test = df_test['id']
y_train = df_train['relevance'].values
x_train = df_train.drop(['id','relevance'],axis=1).values
x_test = df_test.drop(['id','relevance'],axis=1).values
rfr = RandomForestRegressor(n_estimators=1000, random_state=0,max_features = 10, max_depth = 20, n_jobs=-1)
#rfr_bag = BaggingRegressor(rfr, n_estimators=45, max_samples=0.1, random_state=25)
rfr_scores = cross_val_score(rfr,x_train,y_train,scoring = RMSE,cv=4)
print (rfr_scores)
print (rfr_scores.mean())
