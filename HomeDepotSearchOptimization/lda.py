#------------------------------------------------------------------------
# Author : M,Qadri
# Description : Groups the entire dataset into 10 topics
#------------------------------------------------------------------------


from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
import pandas as pd

topic_num=10

#tokenization
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                                
#read the dataset                
docs=pd.read_csv('data_wid_cosine_diff_jac.csv')

#transform the docs into a count matrix
matrix = tf_vectorizer.fit_transform(docs)

#get the vocabulary
vocab=tf_vectorizer.get_feature_names()

#initialize the LDA model
model = lda.LDA(n_topics=topic_num, n_iter=500)

#fit the model to the dataset
model.fit(matrix)

#write the top terms for each topic
top_words_num=20
topic_mixes= model.topic_word_
fw=open('data.txt','w')
for i in range(topic_num):#for each topic
    sorted_indexes=np.argsort(topic_mixes[i])[len(topic_mixes[i])-top_words_num:]#get the indexes of the top-k terms in this topic
    sorted_indexes=sorted_indexes[::-1]#reverse to get the best first    
    my_top=''
    for ind in sorted_indexes:my_top+=vocab[ind]+' ' 
    fw.write('TOPIC: '+str(i)+' --> '+str(my_top)+'\n')
fw.close()


#write the top topics for each doc
top_topics_num=10
doc_mixes= model.doc_topic_
fw=open('data.txt','w')
for i in range(len(doc_mixes)):#for each doc
    sorted_indexes=np.argsort(doc_mixes[i])[len(doc_mixes[i])-top_topics_num:]#get the indexes of the top-k topics in this doc
    sorted_indexes=sorted_indexes[::-1]#reverse to get the best first    
    my_top=''
    for ind in sorted_indexes:my_top+=' '+str(ind)+':'+str(round(doc_mixes[i][ind],2))
    fw.write('DOC: '+str(i)+' --> '+str(my_top)+'\n')
    
fw.close()






