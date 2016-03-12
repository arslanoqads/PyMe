import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


m=Basemap(projection='mill',llcrnrlat=-15,llcrnrlon=25,urcrnrlat=85,
          urcrnrlon=180,resolution='l')


m.drawmapboundary()
m.drawcoastlines()
#m.fillcontinents(color='r',alpha=0.8)
m.drawcountries()
m.drawstates()
#m.drawcounties()
#m.bluemarble()


x,y=[],[]

y11,x11=34.0900,74.7900

x1,y1=m(x11,y11)

m.plot(x1,y1)
x.append(x1)
y.append(y1)

y22,x22=65.0900,-7.7900

x2,y2=m(x22,y22)

m.plot(x2,y2)
#x.append(x2)
#y.append(y2)
#m.plot(x,y,'r',linewidth=2,markersize=15)
#m.drawgreatcircle(x11,y11,x22,y22,linewidth=2,color='c',label='rusva')
#m.etopo()
plt.show()