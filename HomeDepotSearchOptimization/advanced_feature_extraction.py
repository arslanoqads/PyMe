#------------------------------------------------------------------------
# Author : 		M,Qadri
# Description : Contains assorted functions for feature extraction, and 
#				data processing. 
#------------------------------------------------------------------------



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.collocations import *
import nltk
import nltk.tokenize as tk
import  string
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
import enchant
import sklearn.metrics as metrics
from nltk.metrics.distance import jaccard_distance,edit_distance
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.feature_extraction as f
import difflib as dl
import operator
from nltk.corpus import wordnet_ic
import itertools
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
stemmer = SnowballStemmer('english')
cachedStopWords = stopwords.words("english")

#------------------------------------------------------------------------
# Basic Functions
#------------------------------------------------------------------------

# tokenize text
def tokenizer(my_text):
    
    words = nltk.word_tokenize(my_text.decode('utf-8')) 
    word=[i.lower() for i in words]
    return word

# word match count
def query_word_match_count(query,text):
    query = nltk.word_tokenize(query.decode('utf-8')) 

    text = nltk.word_tokenize(text.decode('utf-8')) 
    c=0
    for i in query:
        for j in text:
            if i.lower()==j.lower():
                c+=1
    return c/float(len(query))            
    
# remove stop words from a wordlist
def remove_stopwords(words,cachedStopWords):
    return [word for word in words if word not in cachedStopWords]
        
# get synonyms of a wordlist and add to document. adjacent to original word
def get_synonyms(document):
    n=2
    words =tokenizer(document)

    words=remove_stopwords(words,cachedStopWords)
    morphy=[]
    for i in words:

        word_forms=[i]
        syns=wordnet.synsets(i)
        try:
            wordset=list(set([str(k.name().split('.')[0]) for k in syns][:n]))

        except:
            wordset=[]

        wordset=[j for j in wordset if j not in i]    
        word_forms.extend(wordset)

        morphy.extend(word_forms)
    
    return ' '.join(morphy)

#------------------------------------------------------------------------
# PROCESSING TFIDF
#------------------------------------------------------------------------

syn_query=file1['expanded_query'].apply(get_synonyms)

syn_query1=pd.DataFrame(syn_query)#,columns=['query_with_syn'])
syn_query1.columns=['query_with_syns']


file11=pd.concat([file1, syn_query1], axis=1, join_axes=[file1.index])
# tf-ifildf-------------------------------------------
"""
Step 1 : Initialise model and get TFIDF for corpus.
         Probably a good idea to include query terms in the corpus. Need to check
"""

#initialize 
vect = TfidfVectorizer(max_df=0.9,  stop_words='english', ngram_range=(1,2)) #max_df removes most common words

print 1
#fit
corpus_tfidf=vect.fit_transform(file1.desc)
corpus_voc=vect.vocabulary_

print 2
title_tfidf=vect.fit_transform(file1.title)
title_voc=vect.vocabulary_

print 3
orig_query_tfidf=vect.fit_transform(file1.original_query)
orig_query_voc=vect.vocabulary_

print 4
expanded_query_tfidf=vect.fit_transform(file1.expanded_query)
exp_query_voc=vect.vocabulary_

print 5
syn_query_tfidf=vect.fit_transform(syn_query)#file1.query_with_syns)
syn_query_voc=vect.vocabulary_
print 6

z = corpus_voc.copy()
z.update(title_voc)

z1 = z.copy()
z1.update(orig_query_voc)

z2 = z1.copy()
z2.update(exp_query_voc)

z3 = z2.copy()
z3.update(syn_query_voc)

vocab=z3

v=vocab.keys()

print 7
vect = TfidfVectorizer(max_df=0.9,  stop_words='english', ngram_range=(1,2),vocabulary=v)

print 8

corpus_tfidf=vect.fit_transform(file1.desc)
title_tfidf=vect.fit_transform(file1.title)
print 9
orig_query_tfidf=vect.fit_transform(file1.original_query)
expanded_query_tfidf=vect.fit_transform(file1.expanded_query)
syn_query_tfidf=vect.fit_transform(syn_query)
#syn_query_tfidf=vect.fit_transform(file1.query_with_syns)
print 10
y=cosine_similarity(orig_query_tfidf,corpus_tfidf)

"""
Step 2 : Compare query string with corppus to get cosine distance
"""

#------------------------------------------------------------------------
# COSINE SIMILARITY
#------------------------------------------------------------------------

def cosine_cal(m1,m2):
    scores=[]
    print 'started'
    c=0
    for i in range(m1.shape[0]):
        s=cosine_similarity(m1[i],m2[i])
        scores.append(s)
        c+=1
        if c % 10000 ==0:
            print c,' processed'

    return scores    

#------------------------------------------------------------------------
# TFIDF PROCESSING
#------------------------------------------------------------------------

a1=cosine_cal(orig_query_tfidf,corpus_tfidf)
print 1
a2=cosine_cal(orig_query_tfidf,title_tfidf)
print 2
b1=cosine_cal(expanded_query_tfidf,corpus_tfidf)
print 3
b2=cosine_cal(expanded_query_tfidf,title_tfidf)
print 4
c1=cosine_cal(syn_query_tfidf,corpus_tfidf)
print 5
c2=cosine_cal(syn_query_tfidf,title_tfidf)


a11=pd.DataFrame([i[0][0] for i in a1])
a22=pd.DataFrame([i[0][0] for i in a2])

b11=pd.DataFrame([i[0][0] for i in b1])
b22=pd.DataFrame([i[0][0] for i in b2])

c11=pd.DataFrame([i[0][0] for i in c1])
c22=pd.DataFrame([i[0][0] for i in c2])


tfidf_frames=pd.concat([file1, a11,a22,b11,b22,c11,c22], axis=1, join_axes=[file1.index])

tfidf_frames.columns=  ['none', 'original_query','title','desc','expanded_query', 'query_with_syns', 'orig_query_2_desc','orig_query_2_title','exp_query_2_corpus','exp_query_2_title','syn_query_2_corpus','syn_query_2_title']


del tfidf_frames['none']




orig_query_tfidf=vect.transform(file1.original_query)

def cosine_sim(corpus_tfidf,query):
    #tfidf for query
    query_tfidf = vect.transform([query])
        
    score=[]    
    for i in corpus_tfidf:
        s= (1 - spatial.distance.cosine(query_tfidf.todense(),i.todense()))
        score.append(s)
    return score    

 
#---------------------------------------------
#Jacqard sim
#-------------------------------------------  

def jacquard_sim(text1,text2):
    set1=set(tokenizer(str(text1)))
    set2=set(tokenizer(str(text2)))
    sim=jaccard_distance(set1, set2)#, normalize=True)
    return 1-sim


#-------------------------------------------  
# Diff similarity for closest matches using edit distance
#-------------------------------------------  


def diff_sim(text1,text2):
    sim = dl.get_close_matches    
    s = 0
    wa = text1.split()
    wb = text2.split()
    
    for i in wa:
        if sim(i, wb):
            s += 1
    return float(s) / float(len(wa))

#-------------------------------------------  
# General similarity wrapper
#-------------------------------------------  

def gen_sim(col1,col2,func):
    col1=pd.DataFrame(col1)
    col2=pd.DataFrame(col2)
    c=0
    score=[]    
    for i in range(len(col1)):
        s=func(str(col1.iloc[i]),str(col2.iloc[i]))
        score.append(s)
        c+=1
        if c % 10000 ==0:
            print len(score)
    return score  
 
#------------------------------------------------------------------------
# PROCESSING
#------------------------------------------------------------------------ 
 
#diff
d11=gen_sim(tfidf_frames.original_query,tfidf_frames.title,diff_sim)
print 1
d12=gen_sim(tfidf_frames.original_query,tfidf_frames.desc,diff_sim)
print 2
d21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,diff_sim)
print 3
d22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,diff_sim)
print 4
d31=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.title,diff_sim)
print 5
d32=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.desc,diff_sim)



diff_sim_frames=pd.concat([tfidf_frames, pd.DataFrame(d11),pd.DataFrame(d12),pd.DataFrame(d21),pd.DataFrame(d22),pd.DataFrame(d31),pd.DataFrame(d32)], axis=1, join_axes=[tfidf_frames.index])

diff_sim_frames.columns=  ['original_query','title','desc','expanded_query', 'query_with_syns', 'cosine_orig_query_2_desc','cosine_orig_query_2_title','cosine_exp_query_2_corpus','cosine_exp_query_2_title','cosine_syn_query_2_corpus','cosine_syn_query_2_title','diff_orig2title','diff_orig2desc','diff_exp2title','diff_exp2desc','diff_syn2title','diff_syn2desc']

diff_sim_frames.to_csv('data_wid_cosine_diff.csv')


print 'jacquard'
j11=gen_sim(tfidf_frames.original_query,tfidf_frames.title,jacquard_sim)
print 1
j12=gen_sim(tfidf_frames.original_query,tfidf_frames.desc,jacquard_sim)
print 2
j21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,jacquard_sim)
print 3
j22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,jacquard_sim)
print 4
j31=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.title,jacquard_sim)
print 5
j32=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.desc,jacquard_sim)


jac_sim_frames=pd.concat([tfidf_frames, pd.DataFrame(j11),pd.DataFrame(j12),pd.DataFrame(j21),pd.DataFrame(j22),pd.DataFrame(j31),pd.DataFrame(j32)], axis=1, join_axes=[tfidf_frames.index])

jac_sim_frames.columns=  ['none','original_query','title','desc','expanded_query', 'query_with_syns', 'cosine_orig_query_2_desc','cosine_orig_query_2_title','cosine_exp_query_2_corpus','cosine_exp_query_2_title','cosine_syn_query_2_corpus','cosine_syn_query_2_title','diff_orig2title','diff_orig2desc','diff_exp2title','diff_exp2desc','diff_syn2title','diff_syn2desc','jac_orig2title','jac_orig2desc','jac_exp2title','jac_exp2desc','jac_syn2title','jac_syn2desc']

del jac_sim_frames['none']

jac_sim_frames.to_csv('data_wid_cosine_diff_jac.csv')

tfidf_frames=pd.read_csv('data_wid_cosine_diff_jac.csv')

print 'my_jcn'
jc11=gen_sim(tfidf_frames.original_query,tfidf_frames.title,my_jcn_sim)
print 1
jc12=gen_sim(tfidf_frames.original_query,tfidf_frames.desc,my_jcn_sim)
print 2
jc21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_jcn_sim)
print 3
jc22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_jcn_sim)
print 4
jc31=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.title,my_jcn_sim)
print 5
jc32=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.desc,my_jcn_sim)


jcn_sim_frames=pd.concat([tfidf_frames, pd.DataFrame(jc11),pd.DataFrame(jc12),pd.DataFrame(jc21),pd.DataFrame(jc22),pd.DataFrame(jc31),pd.DataFrame(jc32)], axis=1, join_axes=[tfidf_frames.index])

jcn_sim_frames.columns=  ['none','original_query','title','desc','expanded_query', 'query_with_syns', 
'cosine_orig_query_2_desc','cosine_orig_query_2_title','cosine_exp_query_2_corpus','cosine_exp_query_2_title','cosine_syn_query_2_corpus','cosine_syn_query_2_title',
'diff_orig2title','diff_orig2desc','diff_exp2title','diff_exp2desc','diff_syn2title','diff_syn2desc',
'jac_orig2title','jac_orig2desc','jac_exp2title','jac_exp2desc','jac_syn2title','jac_syn2desc',
'count_orig2title','count_orig2desc','count_exp2title','count_exp2desc','count_syn2title','count_syn2desc',
'lin_orig2title','lin_orig2desc','lin_exp2title','lin_exp2desc','lin_syn2title','lin_syn2desc',
'jcn_orig2title','jcn_orig2desc','jcn_exp2title','jcn_exp2desc','jcn_syn2title','jcn_syn2desc']

del jcn_sim_frames['none']

jcn_sim_frames.to_csv('data_wid_cosine_diff_jac_count_lin_jcn.csv')

tfidf_frames=pd.read_csv('data_wid_cosine_diff_jac_count_lin_jcn.csv')


print 'my_lin'
l11=gen_sim(tfidf_frames.original_query,tfidf_frames.title,my_lin_sim)
print 1
l12=gen_sim(tfidf_frames.original_query,tfidf_frames.desc,my_lin_sim)
print 2
l21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_lin_sim)
print 3
l22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_lin_sim)
print 4
l31=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.title,my_lin_sim)
print 5
l32=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.desc,my_lin_sim)

lin_sim_frames=pd.concat([tfidf_frames, pd.DataFrame(l11),pd.DataFrame(l12),pd.DataFrame(l21),pd.DataFrame(l22),pd.DataFrame(l31),pd.DataFrame(l32)], axis=1, join_axes=[tfidf_frames.index])

lin_sim_frames.columns=  ['none','original_query','title','desc','expanded_query', 'query_with_syns', 'cosine_orig_query_2_desc','cosine_orig_query_2_title','cosine_exp_query_2_corpus','cosine_exp_query_2_title','cosine_syn_query_2_corpus','cosine_syn_query_2_title','diff_orig2title','diff_orig2desc','diff_exp2title','diff_exp2desc','diff_syn2title','diff_syn2desc','jac_orig2title','jac_orig2desc','jac_exp2title','jac_exp2desc','jac_syn2title','jac_syn2desc','count_orig2title','count_orig2desc','count_exp2title','count_exp2desc','count_syn2title','count_syn2desc','lin_orig2title','lin_orig2desc','lin_exp2title','lin_exp2desc','lin_syn2title','lin_syn2desc']

del lin_sim_frames['none']

lin_sim_frames.to_csv('data_wid_cosine_diff_jac_count_lin.csv')

tfidf_frames=pd.read_csv('data_wid_cosine_diff_jac_count_lin.csv')


w11=gen_sim(tfidf_frames.original_query,tfidf_frames.title,query_word_match_count)
print 1
w12=gen_sim(tfidf_frames.original_query,tfidf_frames.desc,query_word_match_count)
print 2
w21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,query_word_match_count)
print 3
w22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,query_word_match_count)
print 4
w31=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.title,query_word_match_count)
print 5
w32=gen_sim(tfidf_frames.query_with_syns,tfidf_frames.desc,query_word_match_count)


count_sim_frames=pd.concat([tfidf_frames, pd.DataFrame(w11),pd.DataFrame(w12),pd.DataFrame(w21),pd.DataFrame(w22),pd.DataFrame(w31),pd.DataFrame(w32)], axis=1, join_axes=[tfidf_frames.index])

count_sim_frames.columns=  ['none','original_query','title','desc','expanded_query', 'query_with_syns', 'cosine_orig_query_2_desc','cosine_orig_query_2_title','cosine_exp_query_2_corpus','cosine_exp_query_2_title','cosine_syn_query_2_corpus','cosine_syn_query_2_title','diff_orig2title','diff_orig2desc','diff_exp2title','diff_exp2desc','diff_syn2title','diff_syn2desc','jac_orig2title','jac_orig2desc','jac_exp2title','jac_exp2desc','jac_syn2title','jac_syn2desc','count_orig2title','count_orig2desc','count_exp2title','count_exp2desc','count_syn2title','count_syn2desc']

del count_sim_frames['none']

count_sim_frames.to_csv('data_wid_cosine_diff_jac_count.csv')

tfidf_frames=pd.read_csv('data_wid_cosine_diff_jac.csv')


o11=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_path_sim)
print 1
o12=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_path_sim)
print 2
o21=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_res_sim)
print 3
o22=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_res_sim)
print 4
o31=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_wup_sim)
print 5
o32=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_wup_sim)
print 6
o41=gen_sim(tfidf_frames.expanded_query,tfidf_frames.title,my_lch_sim)
print 7
o42=gen_sim(tfidf_frames.expanded_query,tfidf_frames.desc,my_lch_sim)

# get word definition

#------------------------------------------------------------------------
# WORD DEFINITION BASED ON POS TAGS
#------------------------------------------------------------------------
def get_definition(word):
    syns = wordnet.synsets(word)
    for s in syns:
        return s.definition()
        

tags=['JJR','JJS','LS','MD','NN','NNS','NNP','NNPS']       
text="And now for something completely different"

#-------------------------------------------  
# replace word in query with its definition
#-------------------------------------------  


def replace_word_with_def(text):
    tags=['JJR','JJS','LS','MD','NN','NNS','NNP','NNPS']  
    it=nltk.pos_tag(nltk.word_tokenize(text))
    sent=[]
    for i in it:
        word=i[0]
        pos=i[1]
        define=[]
        if pos in tags:
            define=str(get_definition(word))
            if str(define)=='None':
                sent.append(word)

                continue;
            sent.append(word)     
            sent.append(define)  
            continue;
        else:
            sent.append(word)
    
    sentence=' '.join(sent)
    return sentence


query_def=file['search_term_spell_checked'].apply(replace_word_with_def)

file1=pd.concat([file1, query_def], axis=1, join_axes=[file1.index])
file1.columns=['original_query','title','desc','expanded_query']



#------------------------------------------------------------------------
# IMPORT CORPUSES FOR JCN, LIN, WCH SIMILARITIES
#------------------------------------------------------------------------

# import corpuses for similarity measures
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

#http://www.nltk.org/howto/wordnet.html
# similarity based on path

#-------------------------------------------  
# JCN similarity
#-------------------------------------------  

def jcn_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        
        sim1=w1.jcn_similarity(w2,semcor_ic)
        sim2=w1.jcn_similarity(w2,brown_ic)

    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score

#-------------------------------------------  
# LIN similarity
#-------------------------------------------  
        
        
def lin_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        

        sim1=w1.lin_similarity(w2,brown_ic) 
        sim2=w1.lin_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score        

#-------------------------------------------  
# PATH similarity
#-------------------------------------------  


def path_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        

        sim1=w1.path_similarity(w2,brown_ic) 
        sim2=w1.path_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score 



#-------------------------------------------  
# LCH similarity
#-------------------------------------------  

def lch_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        

        sim1=w1.lch_similarity(w2,brown_ic) 
        sim2=w1.lch_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score 
#-------------------------------------------  
# WUP similarity
#-------------------------------------------  

def wup_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        

        sim1=w1.wup_similarity(w2,brown_ic) 
        sim2=w1.wup_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score 

#-------------------------------------------  
# RESNIK similarity
#-------------------------------------------  

def res_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        

        sim1=w1.res_similarity(w2,brown_ic) 
        sim2=w1.res_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2)/2
    if score>1:
        return 1
    else:
        return score 
#-----------------------------TEST CODE----------
text1='where are the great desk tables'
text2='where are my great cars'
#-----------------------------TEST CODE----------

#-------------------------------------------  
# Customized JCN similarity
#-------------------------------------------  


#my own version of similarity
def my_jcn_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([jcn_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score
    
#-------------------------------------------  
# CustomizedJ Lin similarity
#-------------------------------------------  
    
def my_lin_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    try:
        sim=pd.DataFrame([lin_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim)  
    except:
        return 0
    return score    
    
#-------------------------------------------  
# Customized JCN wrapper
#-------------------------------------------      
def my_jcn_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([jcn_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score

#-------------------------------------------  
#  CUSTOMIZED PATH wrapper
#------------------------------------------- 

def my_path_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([path_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score

#-------------------------------------------  
#  LCH wrapper
#------------------------------------------- 

def my_lch_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([lch_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score

#-------------------------------------------  
#  Customized WUP similarity
#------------------------------------------- 

def my_wup_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([wup_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score
#-------------------------------------------  
#  Customized Resnik similarity
#------------------------------------------- 

def my_res_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    set1=[i for i in set1 if i not in cachedStopWords and i.isalpha()]
    set2=[i for i in set2 if i not in cachedStopWords and i.isalpha()]
    c_sets=[i for i in itertools.product(set1,set2)]
    
    try:    
        sim=pd.DataFrame([res_sim(i,j) for i,j in c_sets])
        
        sum_score=pd.Series(sim.dropna()[0]).sum()
        
        score=sum_score/len(sim) 
    except:
        return 0
    
    return score    
 