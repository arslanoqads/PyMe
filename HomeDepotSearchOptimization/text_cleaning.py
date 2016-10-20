'''
text cleaning part covers several steps as below:
1) Spelling check, substitute mis-spelled words in queries with correct words, using google spell checking system.
2) Extract brand, color, material information from product_attributes table.
3) Using hand-made word list, derived from data exploration to remove stop words.
4) Tailor-made stemming, eg., substitue certain quantity number and words with unit number.
'''
import pickle
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

#import spell_check_dict
spell_check_dict = pickle.load( open( "spell_check_dict.p", "rb" ) )


df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('attributes.csv')
df_product_description = pd.read_csv('product_descriptions.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})


#remove all the na rows in attributes sheet,creat a new attributes df, df_attr_new
df_attr.dropna(how="all", inplace=True)
df_attr["product_uid"] = df_attr["product_uid"].astype(int)
df_attr["value"] = df_attr["value"].astype(str)

def concate_attrs(df_attr):
    """
    attrs is all attributes of the same product_uid
    """
    names = df_attr["name"]
    values = df_attr["value"]
    pairs  = []
    for n, v in zip(names, values):
        pairs.append(' '.join((n, v)))
    return ' '.join(pairs)
df_attr_new = df_attr.groupby("product_uid").apply(concate_attrs)
df_attr_new = df_attr_new.reset_index(name="product_attributes")

#combine train test sets together, with description, brand, attributes all together
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_product_description, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr_new, how="left", on="product_uid")

#replace missing value in product_attributes and brand with nothing
df_all['product_attributes'] = df_all['product_attributes'].fillna('')
df_all['brand'] = df_all['brand'].fillna('')

# extract material and color
material = dict()
df_attr['about_material'] = df_attr['name'].str.lower().str.contains('material')
for row in df_attr[df_attr['about_material']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    material.setdefault(product, '')
    material[product] = material[product] + ' ' + str(value)
df_material = pd.DataFrame.from_dict(material, orient='index')
df_material = df_material.reset_index()
df_material.columns = ['product_uid', 'material']
color = dict()
df_attr['about_color'] = df_attr['name'].str.lower().str.contains('color')
for row in df_attr[df_attr['about_color']].iterrows():
    r = row[1]
    product = r['product_uid']
    value = r['value']
    color.setdefault(product, '')
    color[product] = color[product] + ' ' + str(value)
df_color = pd.DataFrame.from_dict(color, orient='index')
df_color = df_color.reset_index()
df_color.columns = ['product_uid', 'color']

df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
df_all['material'] = df_all['material'].fillna('')
df_all['color'] = df_all['color'].fillna('')


#spell checking
def spell_check(search):
    if search in spell_check_dict.keys():
        return spell_check_dict[search]
    else:
        return search
df_all['search_term_spell_checked'] = df_all['search_term_spell_checked'].map(lambda x:spell_check(x))


stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"
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

#stemming and remove stop words
start_time = time.time()
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['search_term_spell_checked'] = df_all['search_term_spell_checked'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['product_attributes'] = df_all['product_attributes'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['material'] = df_all['material'].map(lambda x:str_stem(x))
df_all['color'] = df_all['color'].map(lambda x:str_stem(x))


print("--- Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))
