from nltk.tokenize import word_tokenize

#########################################################################
# Author     : Mohamed, Qadri
# Assignment : 2
# Description: Program to count the number negative of words in a data 
#             file, compared to a corpus.
#########################################################################



#define function to read files
def opener(file):
    new_file=set()
    f=open(file)
    for i in f:
        new_file.add(i.lower().strip())
    f.close()
    return new_file    
           
#read file and convert into list, as set is not iteratable           
ip=list(opener('C:\Users\Arslan Qadri\Google Drive\Programs\Python\Web\\gk_class.csv'))
neg=list(opener('C:\\Users\\Arslan Qadri\\Google Drive\\Sem 2\\WebAnalytics\\Assignment 2\\negative-words.txt'))
pos=list(opener('C:\\Users\\Arslan Qadri\\Google Drive\\Sem 2\\WebAnalytics\\Assignment 2\\positive-words.txt'))


#find a list of all words in input
words=[]
for i in ip:
    words.extend(word_tokenize(unicode(i,'utf-8')))
    

#
##select only those negative words that are there in the input    
#neg_words=[i for i in words if i in neg]    
#
##count negative words in the input text
#dict={}
#for i in neg_words:
#    count=0
#    for j in ip:        
#        x=word_tokenize(j)
#        if i in x:
#            count+=1
#    dict[i]=count        
#            
#max_count=[word for word, counts in dict.items() if counts==max(dict.values())]        
#
#
#if len(max_count) == 0:
#    print 'No words found'
#else:
#    print 'The negative word with the highest frequency is : [%s] and the count is %s' % (max_count[0],dict[max_count[0]])
#     
#    
#    
#    
#    
#    
#    
#    