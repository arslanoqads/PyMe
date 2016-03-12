from sklearn.feature_extraction.text import TfidfVectorizer as tv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from nltk.corpus import stopwords, movie_reviews
import nltk
import random


d='An analytics intern at Resource will spend the summer learning and contributing to our fast-paced, challenging and fun agency in the United States on the Mount Everest with Barak Obama in Jammu and Kashmir. You will work with our team to define how we measure our initiatives, and then report on and analyze the results.  You will likely be exposed to a wide variety of analytics tools and initiatives across many digital touchpoints: websites, social media, digital marketing, mobile apps and more.  A passion for analytics and a great willingness to learn are essential.'

#1. tokenize stuff
sents=sent_tokenize(d)
words1=word_tokenize(d)


#2. remove stop words

sw=stopwords.words('english')


#3. stemmming
from nltk.stem import PorterStemmer as ps
from nltk.stem import LancasterStemmer as ls # works same way


p=ps()

stemmed=[]
for i in words1:
    x=p.stem(i)
    stemmed.append(x)


#4. word tagging (POS)
## tag list : https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
#try:
#    tagged1=nltk.pos_tag(words1)
#    
#except Exception as e:
#    print (str(e))    

#5. chunking from sentences
#
#for i in sents:
#    words=word_tokenize(i)
#    tagged=nltk.pos_tag(words)
    #chunky=r""" Chunky : {<DT>*<NN>*<TO>}   
    #                    }<IN><VB|VBP>{            """ #chinking}{  chunking {}
                        
    #chunkParse=nltk.RegexpParser(chunky)
    #chunked = chunkParse.parse(words)
    #chunked.draw()
    #print (chunked)
    
    #namedEntity = nltk.ne_chunk(tagged, binary=True)
    #namedEntity.draw()


#6. copuses

#from  nltk.corpus import gutenberg
#
#sample =gutenberg.raw('bible-kjv.txt')
#
#bible_t=sent_tokenize(sample)


#7. wordnet

#from nltk.corpus import wordnet 


#syn=wordnet.synsets('eat')

#print(syn[0])
#print(syn[0].lemmas()[0].name())
#print(syn[0].name())
#print (syn[0].examples())
#print (syn[0].definition())

antonyms=[]
synonyms=[]
#
#for i in wordnet.synsets('good'):
#    for j in i.lemmas():
#        synonyms.append(j.name())
#        if j.antonyms():
#            antonyms.append(j.antonyms()[0].name())


#8. similarity

#w1=wordnet.synset('ship.n.01')
#w2=wordnet.synset('boat.n.01')

#simi=w1.wup_similarity(w2) #90%


#9. text classification


""" test reviews """
docs=[(list(movie_reviews.words(fileid)),category)  #3 append to list 
for category in movie_reviews.categories()          #1
for fileid in movie_reviews.fileids(category)]      #2


random.shuffle(docs)

""" all word counts """
all_words=[]

""" get list of all words from corpus """
for w in movie_reviews.words():
    all_words.append(w.lower())

""" get count of all words """
all_words=nltk.FreqDist(all_words)

""" take the first 3000 words """
word_feats=list(all_words.keys())[:3000]

""" find those words that exist in the test reviews (doc) 
    [{word :True/False..}, category] """

feats={}
def features(doc):
    words=set(doc)
    for w in word_feats:
        feats[w]=(w in words)
        
    return feats    
    
#print((features(movie_reviews.words('neg/cv000_29416.txt'))))

fsets=[(features(word),cat) for (word,cat) in docs]


training=fsets[:1900]
test=fsets[1900:]

classifier=nltk.NaiveBayesClassifier.train(training)
print 'accuracy :',nltk.classify.accuracy(classifier,test)*100

"to test"

=classifier.classify(features(['its','bad','worst','no','yeah']))











