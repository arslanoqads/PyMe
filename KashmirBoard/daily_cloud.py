import pandas as pd
import ast
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import datetime
import sys
import re

try:
    
 file_in=sys.argv[1] #datefix file


except:
 pass   

img_mask='//images/kashmir.png'


x=pd.read_csv(file_in)
x.created_at=pd.to_datetime(x.created_at)

month=max(x.created_at.dt.month)
m=x[x.created_at.dt.month==month]

today=max(m.created_at.dt.day)
t=m[m.created_at.dt.day.isin([today,today-1])]


ttags=[]
r=[ast.literal_eval(i) for i in t.tags.tolist() if len(i)>2]
for i in r :ttags.extend(i)
#ttags=[i.capitalize() for i in ttags if len(i)>2]
tt=pd.DataFrame(ttags,columns=['txt'])
tt=tt.txt.tolist()
tt=[i for i in tt if i.lower() not in ['sexy','nude','pakistan','india','kashmir']]
txt=' '.join(tt)

txt=txt.encode('utf-8').encode('string-escape')
txt=re.sub('\\\\[\w]+','',txt)



background_color='#EEEEEE'

mask = np.array(Image.open(img_mask))


cloud = WordCloud(max_words=2000, mask=mask, margin=0.1,background_color=background_color, random_state=1).generate(txt)

# store default colored image
default_colors = cloud.to_array()

im = Image.fromarray(default_colors)

im.save('/kashmir_cloud.png')
