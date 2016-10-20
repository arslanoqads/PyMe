import matplotlib.animation as an
import matplotlib.pyplot as plt
import matplotlib.style as style
import twitter_api
import threading
import time
import strtoint

style.use('fivethirtyeight')

hashtags=[['kashmir'],['datascience'],['donaldtrump'],['tesla']]


fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

#C:\\Users\\Arslan Qadri\\myflask\\wsgi\\sentiments\\twitter_tesla.txt

#create fucntion for animation. this will be repeated everytime    
def plot(tag):
    
    file=open('<file>'+tag[0].lower()+'.txt','r').read()
    
    lines=file.split('\n')
    
    x1=[]
    y1=[]
    
    x,y=0,0
    for o in lines:
        x+=1
        if 'pos' in o:
            y+=1
        elif 'neg' in o:
            y-=1
            
        x1.append(x)
        y1.append(y)

    ax1.clear()
    ax1.plot(x1,y1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_title(tag[0]+' Tweet Sentiment Graph',color='black',fontsize=11,y=0.95,x=0.750,bbox=dict(facecolor='white', edgecolor='black'))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sentiment')    
    x=ax1.get_yticks()
    
    x=strtoint.StrToInt(str(x)[1:-1],'.')
    
    try:
        zindex=x.index(0)
    except:
        x.append(0)
        x.sort()
        zindex=x.index(0)
        
    x=['' for i in x]
    x[-1]='Good'        
    x[0]='Bad'
    x[zindex]='Neutral'
    
    ax1.set_yticklabels(x)
    fig.tight_layout()
    
    
    plt.savefig('<file>/twitter_'+tag[0].lower()+'.png')    

for i in hashtags:
     
    plot(i)
    time.sleep(10)    
         
    
