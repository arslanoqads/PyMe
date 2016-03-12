import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as d
import datetime as dt
import datetime
import numpy as np
import column_trim as ct
import matplotlib.style as s
import matplotlib.ticker as mticker

s.use('fivethirtyeight')



#d3=d1[d1['longitude'] > 25]
#d3=d3[d3['latitude']  >-15]
#d3=d3[d3['longitude'] < 180]
#d3=d3[d3['latitude']  < 85]




#map = Basemap(projection='mill')#,llcrnrlat=-15,llcrnrlon=25,urcrnrlat=85,
          #urcrnrlon=180,resolution='l')

          
#Basemap(projection='mill',urcrnrlat=0,urcrnrlon=0,resolution='l')

#map.drawcoastlines()
#map.drawcountries()
#map.fillcontinents(color = 'coral')
#map.drawmapboundary()

#ct.trim(d1,'time',0,19)

#lon=list(d3['longitude'])
#lat=list(d3['latitude'])  
 
#lons = [-1, -134.8331, -134.6572]
#lats = [57.0799, 57.0894, 56.2399]
 
 
#x,y = map(lon,lat)
#map.plot(x, y, 'bo', markersize=1)
# 
 
#plt.plot(d1['mag']) 
#plt.plot_date(d1['time'],d1['mag']) 
# 

                        
d1=pd.read_csv('temp1.csv')

d1.sort('time')
a1=plt.subplot(111)
a1.plot_date(d1['time'],d1['depth'],linestyle='-') 
a1.axes.xaxis.set_ticklabels(d1['place'])

for lab in a1.xaxis.get_ticklabels():
    lab.set_rotation(45)

plt.show()   

for j in c:
    k=len([i for i in x if i==j])
    y.append((j,k))
    #pd.unique([i for i in x if i=='the'])

#seperate code
a1=plt.subplot(111)
plt.plot(range(len(x2)),x2)
a1.axes.xaxis.set_ticklabels(x1)
for lab in a1.xaxis.get_ticklabels():
    lab.set_rotation(45)

plt.show() 
