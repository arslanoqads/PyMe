#!/usr/bin/env python2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import pos_tag
import re



#input data
#file= open('jobs.txt','r')
image_path='C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\temp_codes\\image.jpg'
background_color='white'

# read the mask image
mask = np.array(Image.open(image_path))



#data cleaning and processing
f1=[]
for line in file:
    line=line.strip()
    line=filter(None,re.split('[],.-/\\_ ]+',line))  #splitting based on multiple delimiters    
    for i in line:
        f1.append(i.lower())
        


# remove punctuations and get only nouns/ adjectives
all_words=[]
swords=stopwords.words()
all_words=[i for i in f1 if i.isalpha()]    
all_words=[i for i in all_words if i not in swords]  
all_nouns=[w for w,t in pos_tag(all_words) if t in ['NN','NNS','NNP','NNPS','JJ','JJR','JJS']]
final_set=' '.join(all_nouns)



########################################################
# for testing

text=''
text = open("ahope.txt").read()

########################################################


cloud = WordCloud(max_words=20000, mask=mask, margin=0.1,
               background_color=background_color,
               random_state=2).generate(text)

# store default colored image
default_colors = cloud.to_array()

#plt.figure()
#plt.show()
plt.title("Data Cloud")
plt.imshow(default_colors)
plt.axis("off")
plt.savefig('C:\\Users\\Arslan Qadri\\Google Drive\\Programs\\Python\\temp_codes\\2.jpg',dpi=4000) 
#plt.show()