from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import path

#Convert all the required text into a single string here 
#and store them in word_string

#you can specify fonts, stopwords, background color and other options



#mask image
d='http://www.stencilry.org/stencils/movies/star%20wars/storm-trooper.gif'


mask = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))

word_string='arslan ars is the arslan is the'

wordcloud = WordCloud().generate(word_string)
"""
wordcloud = WordCloud(#font_path='/Users/kunal/Library/Fonts/sans-serif.ttf',
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(word_string)
"""

plt.imshow(wordcloud)
plt.axis('off')
plt.show()