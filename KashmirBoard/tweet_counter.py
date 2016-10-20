
import pandas as pd

import sys
import time
import copy
#cleaning csv

try:
 file_in=sys.argv[1] #kashmir_tweet_pd
 file_out=sys.argv[2]  #smoothen_file
 timeline=sys.argv[3]  #timeline file 
except:
 pass   

# reads tweet file and appends the count of hourly tweets to master file
def count_extractor(file_in,file_out):




    df=pd.read_csv(file_in)
    df.columns=['text','verified','screen_name','created_at','entities','name','followers_count','location','time_zone','friends_count','utc_offset']
    
    
    df=df[['verified','created_at','screen_name']]
    
    y=df[df.verified.isin(['True','TRUE','FALSE','False',True,False])]
            
    y.created_at=y.created_at.apply(lambda x : x if str(x)[-3:-1]=='01' else '')
    y.created_at=pd.to_datetime(y.created_at,infer_datetime_format=True)
     
    y.created_at=y.created_at+pd.Timedelta(hours=5.5)
     
    y.created_at=y.created_at.apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))  
    y.created_at=pd.to_datetime(y.created_at)
    y.index=y.created_at    
    

    
    
    z=y.resample('60T',how='count')  
    z=z['screen_name']
    z.to_csv(file_out,encoding='utf-8',header=False,mode='a')







def plotly_intializer(file_out,timeline):
    
    
    df=pd.read_csv(file_out)
    df.drop_duplicates(inplace=True)
    
    df= df[['screen_name','created_at']]
    df.index=df.created_at
    del df['created_at']
    
 
    
    #del df['created_at']
    df=pd.rolling_mean(df,window=10,min_periods=1)
    hour = df.index
    
    df.reset_index(inplace=True)
    df.index=df.created_at
    
    
    df.created_at=pd.to_datetime(df.created_at)    
    dt=pd.read_csv(timeline)
    
    dt=dt[2:]
    dt.date=pd.to_datetime(dt.date,infer_datetime_format=True)
    
    dt.date=dt.date.apply(lambda x : str(x).split(' ')[0]+' 12:00:00')
    dt.date=pd.to_datetime(dt.date,infer_datetime_format=True)    
    
    #for events    
    l=[]
    for i in dt.index:
      try: 
       if str(dt.event.ix[i])!='nan':
        d={}   
        d['arrowhead']=str(list(df[df.created_at==dt.ix[i].date].screen_name)[0])
        d['ax']=0
        d['ay']=-10
        d['showarrow']=True
        d['text']=dt.event.ix[i]
        d['x']=dt.date.ix[i]
        d['xref']='x'
        d['y']=list(df[df.created_at==dt.ix[i].date].screen_name)[0]+2000
        d['y']=d['y']+dt.height.ix[i]
        d['yref']='y'
        #d['bordercolor']='#c7c7c7'
        d['borderwidth']=1
        d['borderpad']=1
        d['bgcolor']='#ffffff'
        d['opacity']=0.85
        d['font']=dict(family='Helvetica',size=9)
        l.append(d)
      except :
        pass   

       
    # Create and style traces
    del df['created_at'] 
    
    bf= pd.rolling_mean(df,window=60,min_periods=1)
    trace0 = go.Scatter(
        x = bf.index.tolist(),
        y = bf.screen_name.tolist(),
        name='Trend',
        line = dict(
            color = 'red',
            width = 1)
    )
    
    trace1 = go.Scatter(
        x = hour,
        y = df.screen_name.tolist(),
        name='Tweets',
        line = dict(
            color = '#07C0CD',
            width = 1.5)
    )
    
    
      
    
    
    trace2 = go.Bar(
        x=dt.date.tolist(),
        y=dt.dead.apply(lambda x: 0 if str(x)=='nan' else x).tolist(),
        name='Civilians Killed',
        yaxis='y2',
        opacity=0.6,
       marker=dict(
        color='000000',
    
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=0.1),
    ),
    #name=''
    #orientation='h'
    )    
    trace3 = go.Bar(
        x=dt.date.tolist(),
        y=dt.injured.apply(lambda x: 0 if str(x)=='nan' else x).tolist(),
        name='Civilians Injured',
        yaxis='y3',
        opacity=0.6,
       marker=dict(
        color='red',
    
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=0.1),
    ),
    #name=''
    #orientation='h'
    )    
       
    

    fig = tools.make_subplots(rows=3, cols=1,shared_xaxes=True)
    
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)    
    
    fig['layout'].update(annotations=l,title = 'Daily Tweets on #Kashmir',
                 xaxis=dict(autotick=True,showgrid=False,
                            zerolinewidth=2,
                            rangeslider=dict()),
              
            yaxis=dict(domain=[0.50, 1],autorange=True,title='Tweet Count'),
           yaxis2=dict(domain=[0.25, 0.4],autorange=True,showgrid=False,title='Civilian Deaths'),
          yaxis3=dict(domain=[0, 0.15],autorange=True,showgrid=False,title='Civilian Injuries'))
    
    #fig = dict(data=data, layout=layout)
    #py.sign_in('arazalan','xiaimi0443')
    py.plot(fig,filename='styled-line')




count_extractor(file_in,file_out)
time.sleep(2)

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

#py.sign_in('arslanoqads', 'ruuzqpx3lo')
py.sign_in('arazalan','xiaimi0443')
plotly_intializer(file_out,timeline) 
