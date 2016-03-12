import pandas as pd
import datetime 
import matplotlib.dates as d
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import copy
import seaborn as sns
import matplotlib.style as s

s.use('fivethirtyeight')

def trim(frame,column,start,end):    
    i=0
    while i<len(frame):
        frame[column][i]=frame[column][i][start:end]
        i=i+1

#map = Basemap(projection='mill')
#map.drawcoastlines()
#map.drawcountries()
#map.fillcontinents(color = 'coral')
#map.drawmapboundary()

d1=pd.read_csv("C:\Users\Arslan Qadri\Downloads\\clean_quakes.csv")

#d1=d1.head(1000)
d1.sort('type',inplace=True)

def plot_map(lon,lat,color,label):              
    x,y = map(lon,lat)
    map.plot(x, y, 'bo',color=color, markersize=3,label=label)

types=pd.Series(['explosion', 'nuclear explosion', 'mine collapse',
       'rock burst', 'quarry blast', 'rockslide', 'quarry',
       'chemical explosion', 'mining explosion', 'landslide', 'sonic boom',
       'anthropogenic event', 'acoustic noise'])

colors=['blue','green','red','cyan','magenta','yellow','black','blue','green','red','cyan','magenta','yellow','black','red','cyan','magenta','yellow']

labels=[]
counts=[]
dict=[]
for x in types:
    c=colors.pop()
    x1=x.replace(' ','_')
    y=x1
    temp=d1[d1['type']==x]
    #plot_map(list(temp['longitude']),list(temp['latitude']),c,x1)
    #plt.bar(range(len(types)),len(temp),label=x1)    
    counts.append(len(temp))
    labels.append(x1.title().replace('_',' '))
    dict.append((x1,len(temp)))
    exec(y+'=temp')


bar=pd.DataFrame(dict)


    
ax1=plt.subplot2grid((1,1),(0,0))

ax1.bar(range(len(bar)),bar[1],align='center')
ax1.set_xticks(range(len(bar)))
ax1.set_xticklabels(labels)
#ax1.axes.xaxis.set_ticklabels(labels,range(13))
ax1.set_title('Tremors for reasons other than earthquakes (1900-2016)',fontsize=15)
ax1.set_xlabel('Tremor Causes',fontsize=12)
ax1.set_ylabel('Number in the past century',fontsize=12)
for lab in ax1.xaxis.get_ticklabels():
    lab.set_rotation(90)         

for x,y in zip(range(len(bar)),bar[1]):
    plt.text(x,y,bar[1][x],ha='center', fontsize=10)
plt.show() 
 


