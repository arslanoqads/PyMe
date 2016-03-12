import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mt
import matplotlib.finance as mf

import matplotlib.dates as dt
import datetime
import matplotlib.pyplot as plt
import time

#create data
y=[1,3,4,2,5,5,6,4,5,6,3,4,6,3,4,5]
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
w=[3,2,4,5,2,4,5,5,6,3,4,5,6,2,3,4]
z=[2,3,4,5,3,2,5,5,7,5,6,7,3,4,0,9]
v=[9,8,7,1,2,3,4,9,8,7,2,5,6,7,4,5]
u=[0,1,9,2,3,0,2,9,3,8,4,9,8,4,5,6]
d=[ 735925,735926,735927,735929,735932,735933,735934,735935,
 735936,735939,735940,735941,735942,735943,735946,735947,]

#change into array format, otherwise plotting has problem
x=np.array(x)
y=np.array(y)

#concatenate for the candle plot
l=[]
i=0
while i<=15:
    a=d[i],y[i],w[i],z[i],v[i],u[i]
    l.append(a)
    i+=1

#graph formatting
fig=plt.figure()
ax1=plt.subplot2grid((1,1),(0,0))
ax1.grid(True,linewidth=0.04,linestyle='-')
ax1.set_yticks(x)
ax1.set_xticks(x)

#candle plot and date formatting
mf.candlestick_ohlc(ax1,l,width=0.1,colorup='g',colordown='r')
ax1.xaxis.set_major_locator(mt.MaxNLocator(10))
ax1.xaxis.set_major_formatter(dt.DateFormatter('%Y-%m-%d'))

#annotate text at a given location with text. 
ax1.annotate('Fire!!', (d[5],w[6]), xytext=(0.1,0.8),
             arrowprops=dict(facecolor='r',color='c'),textcoords='axes fraction')

#annotate with shapes
bboxprop=dict(boxstyle='round4,pad=0.4',fc='y',ec='b')             
ax1.annotate(str(w[-1]),(d[-1],w[-1]),xytext=(d[-1],w[-1]),bbox=bboxprop)

for lab in ax1.xaxis.get_ticklabels():
    lab.set_rotation(45)

#plt.fill_between(x,y, facecolor='c',alpha=0.5)

plt.show()
