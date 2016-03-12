import matplotlib.animation as an
import matplotlib.pyplot as plt


#declare area
fig=plt.figure()
ax1=fig.add_subplot(1,1,1) 

#create fucntion for animation. this will be repeated everytime    
def anim(i):
    file=open('test.txt','r').read()
    x1=[]
    y1=[]
    data=file.split('\n')
    for ox in data:
        x,y=ox.split(',')
        x1.append(x)
        y1.append(y)

    ax1.clear()
    ax1.plot(x1,y1)

#standard function to call animation
ani=an.FuncAnimation(fig,anim,interval=1000)

plt.show()