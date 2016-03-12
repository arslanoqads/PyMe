import pandas as pd
import datetime
import matplotlib.dates as d
import matplotlib.pyplot as plt
import matplotlib.style as s
from scipy import interpolate as ip
import numpy as np

#use style
s.use('ggplot')

#location of the file
location='C:\Users\Arslan Qadri\Google Drive\Programs\Python\Web\\'

#filename : Lables
pages={'gk':'Greater Kashmir','nyt':'NY Times','ht':'Hindustan Times'}


#read data and extract relevant fields
#correct date format
def clean_data(file):
    data=pd.read_csv(file)
    data=pd.DataFrame(data,columns=['status_id','link_name','status_type','status_published','num_likes','num_comments','num_shares'])    
   # data=data.head(118)
    rep= lambda y : (y.replace('/','-'))
    date_conv = lambda x : (d.date2num(datetime.datetime.strptime(x,"%m-%d-%Y %H:%M")))
    #real_date = lambda x : (datetime.datetime.strptime(x, "%m-%d-%Y %H:%M").strftime("%m-%d-%Y %H:%M"))
    data['status_published']=data['status_published'].apply(rep)
    data['status_published']=data['status_published'].apply(date_conv)
    
    return data


#data smoothing    
def smoothing(x,y,factor):
    x=list(x)
    y=list(y)
    f = ip.interp1d(x, y, kind="linear")
    x_int = np.linspace(x[0],x[-1], factor)
    y_int = f(x_int)
    return x_int, y_int    

#plot data
#change above file names to dataset variables
#remove outliears-maxlim
def plotter(y_var,line_style,max_lim):
    p=-1
    for i,q in zip(pages.keys(),pages.values()):
        
        url=location+i+'.csv'
        j=i
        k=clean_data(url)
        exec(j+'=k')
        p+=1
        k[k[y_var]>=max_lim]=max_lim
        x,y=smoothing(k['status_published'],k[y_var],smoothing_factor)
        co=c.pop()
        print co,q
        a[p].plot_date(x, y, marker='',linewidth=2,linestyle=line_style,color=co,label=q)

    

smoothing_factor=60



fig = plt.figure()
################################################
#plot likes
a=[]
a.append(fig.add_subplot(311))
for s in range(1,6):
    a.append(a[0].twinx())
    a[s].axes.get_yaxis().set_visible(False)      
a[1].get_shared_y_axes().join(a[0],a[1],a[2],a[3],a[4],a[5])
#a[0].axes.get_xaxis().set_visible(False) 
a[0].set_xticklabels(['','','','','','','','','','','','','','','','','','','','','','','','',])
temp=a[0]
y_var='num_likes'
line_style='-'
c=['white','cyan','red']
plotter(y_var,line_style,18000)

l=range(1,8)
h=range(1,8)
for x in range(1,7):
    h[x],l[x] =a[x-1].get_legend_handles_labels()
        
for lab in a[0].xaxis.get_ticklabels():
    lab.set_rotation(45)

a[0].legend(h[1]+h[2]+h[3]+h[4]+h[5]+h[6],l[1]+l[2]+l[3]+l[4]+l[5]+l[6],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,fontsize=10,fancybox=True)             
a[0].set_title('Likes per post',fontsize=7,y=0.8,x=0.90,bbox=dict(facecolor='white', edgecolor='black'))
#a[0].set_ylabel('Counts',fontsize=12)   

################################################
#plot shares
a=[]
a.append(fig.add_subplot(312))
for s in range(1,6):
    a.append(a[0].twinx())
    a[s].axes.get_yaxis().set_visible(False)      
a[1].get_shared_y_axes().join(a[0],a[1],a[2],a[3],a[4],a[5])
a[0].set_xticklabels(['','','','','','','','','','','','','','','','','','','','','','','','',])

y_var='num_shares'
line_style='-'
c=['white','cyan','red']
plotter(y_var,line_style,4000)

l=range(1,8)
h=range(1,8)
for x in range(1,7):
    h[x],l[x] =a[x-1].get_legend_handles_labels()
        
for lab in a[0].xaxis.get_ticklabels():
    lab.set_rotation(45)

#a[0].legend(h[1]+h[2]+h[3]+h[4]+h[5]+h[6],l[1]+l[2]+l[3]+l[4]+l[5]+l[6],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,fontsize=10)             
a[0].set_title('Shares per post',fontsize=7,y=0.8,x=0.90,bbox=dict(facecolor='white', edgecolor='black'))
#a[0].set_ylabel('Counts',fontsize=12)   

################################################
#plot comments
a=[]
a.append(fig.add_subplot(313))
for s in range(1,6):
    a.append(a[0].twinx())
    a[s].axes.get_yaxis().set_visible(False)      
a[1].get_shared_y_axes().join(a[0],a[1],a[2],a[3],a[4],a[5])


y_var='num_comments'
line_style='-'
c=['white','cyan','red']
plotter(y_var,line_style,600)


l=range(1,8)
h=range(1,8)
for x in range(1,7):
    h[x],l[x] =a[x-1].get_legend_handles_labels()
        
for lab in a[0].xaxis.get_ticklabels():
    lab.set_rotation(0)

#a[0].legend(h[1]+h[2]+h[3]+h[4]+h[5]+h[6],l[1]+l[2]+l[3]+l[4]+l[5]+l[6],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,fontsize=10)             
a[0].set_title('Comments per post',fontsize=7,y=0.8,x=0.90,bbox=dict(facecolor='white', edgecolor='black'))
a[1].set_xlabel('Data Source Facebook Graph',fontsize=8,x=0.92,y=-0.1)           
a[0].set_xlabel('Newspaper Metrics 2013-Present. Source Facebook Graph',fontsize=8,x=0.850,y=-1.5)  
plt.show()
    

#resamp = df.set_index('date').groupby('string').resample('M', how='sum')