import nltk
from nltk.collocations import *


def bigrams(my_text):
    my_text=my_text.decode('ascii','ignore')
    words = nltk.word_tokenize(my_text)
    words=[i for i in words if   i.isalpha()]
    my_bigrams = nltk.bigrams(words)
    return [i for i in my_bigrams]



def trigrams(my_text):
    my_text=my_text.decode('ascii','ignore')
    words = nltk.word_tokenize(my_text)
    words=[i for i in words if   i.isalpha()]
    my_trigrams = nltk.trigrams(words)
    return [i for i in my_trigrams]


def ngrams(my_text,n):
    my_text=my_text.decode('ascii','ignore')
    words = nltk.word_tokenize(my_text)
    words=[i for i in words if   i.isalpha()]
    my_ngrams = nltk.ngrams(words,n)
    return [i for i in my_ngrams]
    
    
def my_bigram(my_text,n):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    my_text=my_text.decode('ascii','ignore')
    words = nltk.word_tokenize(my_text)
    words=[i for i in words if   i.isalpha()]
    finder = BigramCollocationFinder.from_words(words)  
    word_set=finder.nbest(bigram_measures.pmi, n) 
    return word_set


def my_trigram(my_text,n) :
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    my_text=my_text.decode('ascii','ignore')
    words = nltk.word_tokenize(my_text)
    words=[i for i in words if   i.isalpha()]  
    finder = TrigramCollocationFinder.from_words(words)  
    word_set=finder.nbest(trigram_measures.pmi, n) 
    return word_set



  