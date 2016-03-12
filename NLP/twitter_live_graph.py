import matplotlib.animation as an
import matplotlib.pyplot as plt
import matplotlib.style as style
import twitter_api
import threading
import time
import strtoint

style.use('fivethirtyeight')

tag=['Jnu']

class getdata(threading.Thread):  
    def run(self):
        try:
            print 'in getdata',tag
            twitter_api.caller(tag)
            print 'out of get data'
        except:
            time.sleep(2)
#declare area
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)


#create fucntion for animation. this will be repeated everytime    
def anim(i):
    file=open('sentiments\\twitter_'+tag[0]+'.txt','r').read()
    
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
    ax1.set_title(tag[0]+' Tweet Sentiment Graph')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sentiment')    
    x=ax1.get_yticks()
    
    x=strtoint.StrToInt(str(x)[1:-1],'.')
    

    zindex=x.index(0)
    x=['' for i in x]
    x[zindex]='Neutral'
    x[-1]='Happy'
    x[0]='Sad'

    
    ax1.set_yticklabels(x)

#standard function to call animation
ani=an.FuncAnimation(fig,anim,interval=5000)       

        
x=getdata()


#trigger the threads
x.start()
#plt.legend()
plt.show()