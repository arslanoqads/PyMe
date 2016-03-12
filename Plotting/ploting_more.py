import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.style as s

s.use('fivethirtyeight')
                        
d1=pd.read_csv('temp1.csv')

d1.sort('time')


#create subplots        
a1=plt.subplot(111)
#a2=plt.subplot(312)  
#a3=plt.subplot(312)

#plot the subplots        
a1.plot_date(d1['time'],d1['depth'],linestyle='-') 
#a2.plot_date(d1['time'],d1['mag'],linestyle='-') 
#a3.plot(d1['place'],d1['dept'])


#ticks management
#a1.set_xticks(range(len(d1)))
#plt.xticks(range(len(d1)),d1['place'])
a1.axes.xaxis.set_ticklabels(d1['place'])


#rotation
#for lab in a1.xaxis.get_ticklabels():
#   lab.set_rotation(45)

#rotate
for lab in a1.xaxis.get_ticklabels():
    lab.set_rotation(45)
#plt.savefig('abc.jpg')
plt.show()   
            