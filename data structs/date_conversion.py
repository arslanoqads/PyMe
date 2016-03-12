import matplotlib.dates as d
import datetime as dt
import datetime
import numpy as np

import time


#Convert from date format to number formal

str_date='20140205'  #date format

#Change Date format
temp=datetime.datetime.strptime(str_date, '%Y%m%d').strftime('%Y-%m-%d')
#convert 'temp' to datetime format
l_date=datetime.datetime.strptime(temp,'%Y-%m-%d')
#Convert to number
l_num=d.date2num(l_date)

print "The derived number is :",l_num



#Convert from number format to date

num_date=735992.321123      #number format  
#Converson function
date_type=d.num2date(num_date)
#Print
print "The date is :",date_type



#Convert from number to time format
    
tx='1435325420'
tx=1436027153.0
tm=int(tx)

t_time=datetime.datetime.fromtimestamp(tm)  
print "This is is the time : ", t_time


#Convert from time format to number
r_time='2015-06-26 00:00:00'
pattern = '%Y-%m-%d %H:%M:%S'

epoch = int(time.mktime(time.strptime(r_time, pattern)))
print "The number format is",epoch


