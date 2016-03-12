import urllib
import numpy  as np
import matplotlib.dates as dt
import datetime
import matplotlib.pyplot as plt
import time


def coco(str_date,fmt):
        
    #Change Date format
    temp=datetime.datetime.strptime(str_date, fmt).strftime('%Y-%m-%d')
    #convert 'temp' to datetime format
    l_date=datetime.datetime.strptime(temp,'%Y-%m-%d')
    
    l_num=dt.date2num(l_date)
    
    return l_num

def momo(tim):
    tm = np.vectorize(datetime.datetime.fromtimestamp) 
    return tm(tim)

url='http://chartapi.finance.yahoo.com/instrument/1.0/tsla/chartdata;type=quote;range=1y/csv'

source=urllib.urlopen(url).read()


#initiate the plot
fig=plt.figure()
ax1=plt.subplot2grid((1,1),(0,0))

#splitting from data
rows=[]
x=source.split('\n')

for row in x:
    y=row.split(',')
    if 'close' not in y and len(y)==6:
        rows.append(y)
 
date=[]
close=[]
high=[]
low=[]
opn=[]
vol=[]  
 

#unpacking and changing the date format   
for row in rows:
    a,b,c,d,e,f=row
   # date.append(a)
    date.append(coco(a,'%Y%m%d'))    
    close.append(b)
    high.append(c)
    low.append(d)
    opn.append(e)
    vol.append(f)

#formating data, there was some problem in data format
data=[]
for tm in date:
    t_time=datetime.datetime.fromtimestamp(float(tm))  
    print "This is is the time : ", t_time
    data.append(t_time)

#use this to plot with date
plt.plot_date(data,high,'-')

#use this to color
plt.fill_between(low,close,240)


#format graph
ax1.grid(True,linewidth=1,linestyle='-')
plt.xlabel('Time')
plt.ylabel('Price')
ax1.yaxis.label.set_color('m')
ax1.xaxis.label.set_color('b')

for lab in ax1.xaxis.get_ticklabels():
    lab.set_rotation(45)
    
plt.show()