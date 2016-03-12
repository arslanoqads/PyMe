import matplotlib.pyplot as plt
import random
import matplotlib.ticker as mticker
fig=plt.figure()

#create random numbers
def create():
    xs=[]
    ys=[]
    
    for i in range(10):
        y=random.randrange(8)
        
        xs.append(i)
        ys.append(y)
    return xs, ys    
    
    
x1,y1 = create()
x2,y2 = create()
x3,y3 = create()    


#ax1=fig.add_subplot(211)
#ax1.plot(x1,y1)

#ax1=plt.subplot2grid((6,1),(0,0),rowspan=1, colspan=1)

#######################################################

#ax2=fig.add_subplot(223)
#ax2.plot(x2,y2)

#ax2=plt.subplot2grid((6,1),(3,0),rowspan=2, colspan=1)

#######################################################

#ax3=fig.add_subplot(224)
ax3=plt.subplot2grid((6,1),(0,0),rowspan=6, colspan=1)


#plt.setp(ax3.get_yticklabels(),visible=False)

#superimpose using twinx
ax3b=ax3.twinx()
ax3c=ax3.twinx()

#change bin settings
ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4,prune='lower'))

ax3b.fill_between(x2,y2,facecolor='b',alpha=0.3)
ax3.grid(True)
ax3.plot(x3,y3,label='legend')
ax3c.plot(x1,y1)

ax3b.set_ylim(0,40)
#ax3b.axes.yaxis.set_ticklabels([])
leg=ax3.legend(loc=9,ncol=2, prop={'size':11}, fancybox=True, borderaxespad=0)
leg.get_frame().set_alpha(0.4)

plt.show()