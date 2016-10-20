
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd


import datetime
x=pd.read_csv('clean_datefix3.csv')
x.created_at=pd.to_datetime(x.created_at)


month=max(x.created_at.dt.month)
week=max(x.created_at.dt.week)





m=x[x.created_at.dt.month==month]
w=x[x.created_at.dt.week==week]

today=max(x.created_at.dt.day)
t=m[m.created_at.dt.day.isin([today,today-1])]


mtags=[]
for i in m.tags.tolist():mtags.extend(i)
mtags=[i.capitalize() for i in mtags if len(i)>2]
mt=pd.DataFrame(mtags,columns=['txt'])
mcounts=pd.DataFrame(mt.txt.value_counts())
mcounts=mcounts[mcounts.index!='Kashmir'].head(50)
#mcounts.to_csv(month+'month.csv')



wtags=[]

r=[wtags.extend(ast.literal_eval(i)) for i in w.tags.tolist() if len(i)>2]
wtags=[i for i in wtags if i.lower() not in ['sexy','nude','kashmir']]
wtags=[i.capitalize() for i in wtags if len(i)>2]
wt=pd.DataFrame(wtags,columns=['txt'])
wcounts=pd.DataFrame(wt.txt.value_counts())
wcounts=wcounts[wcounts.index!='Kashmir'].head(50)


ttags=[]
r=[ast.literal_eval(i) for i in t.tags.tolist()]
for i in r :ttags.extend(i)
ttags=[i.capitalize() for i in ttags if len(i)>2]
tt=pd.DataFrame(ttags,columns=['txt'])
tt=tt.txt.tolist()
#tcounts=pd.DataFrame(tt.txt.value_counts())
#tcounts=tcounts[tcounts.index!='Kashmir']

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
    
leftMin=tcounts.txt.min()
leftMax=tcounts.txt.max()

tcounts.reset_index(inplace=True)
tcounts.index=tcounts['index'].apply(lambda x: x.encode('utf-8'))

del tcounts['index']
tcounts.txt=tcounts.txt.apply(lambda x : translate(x, leftMin, leftMax, 10, 20) )

d=[i for i in zip(tcounts.index,tcounts.txt)]
e=[list(i) for i in d if i[0].encode('string-escape')[0]!='\\' and i[0]!='Nude']




######################################################
#month

def plotly_monthtags(mcounts):
    ml=mcounts.index.tolist()[:25]
    mln=mcounts.txt.tolist()[:25]

    ml.reverse()
    mln.reverse()    
    trace0 = go.Bar(
        y = ml,
        x = mln,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='',
    orientation='h'
    )

    
    data = [trace0]
    
    # Edit the layout 
    layout = dict(
    title = 'Hot topics on Kashmir this September',
    xaxis = dict(title = 'Impressions'),
    
        margin=dict(
        l=200,
        r=50,
        t=70,
        b=70
    ))
                  
    
    # Plot and embed in ipython notebook!
    # options in the graph such as displaying edit text and other tools 
    # can be hidden in iframe options in HTMS
    fig = dict(data=data, layout=layout)
    py.iplot(fig,filename='styled-line')



py.sign_in('xxx', 'xxx')
plotly_monthtags(mcounts) 

######################################################
#week

def plotly_weektags(wcounts):
    ml=wcounts.index.tolist()[:25]
    mln=wcounts.txt.tolist()[:25]

    ml.reverse()
    mln.reverse()    
    trace0 = go.Bar(
        y = ml,
        x = mln,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='',
    orientation='h'
    )

    
    data = [trace0]
    
    # Edit the layout 
    layout = dict(
    title = 'Hot topics this week on Twitter',
    xaxis = dict(title = 'Impressions'),
    
        margin=dict(
        l=200,
        r=50,
        t=70,
        b=70
    ))
                  
    
    # Plot and embed in ipython notebook!
    # options in the graph such as displaying edit text and other tools 
    # can be hidden in iframe options in HTMS
    fig = dict(data=data, layout=layout)
    py.iplot(fig,filename='styled-line')

import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('xxx', 'xxx')
plotly_weektags(wcounts) 


######################################################
#month
