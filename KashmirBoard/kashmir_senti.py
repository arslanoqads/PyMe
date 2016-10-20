import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import os
import time


def sentiment_gauge(sentiment_score):
    
    base_chart = {
        "values": [20, 20, 20],
        "labels": ["Sad", "Neutral", "Happy"],
        "domain": {"x": [0, .48]},
        "marker": {
                'colors': [
                
                '#FF6961',
                '#FFEF00',
                '#03C03C',
                'rgb(255,255,255)'
               
            ],
            "line": {
                "width": 1
            }
        },
        "name": "Gauge",
        "hole": .65,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 190+sentiment_score*(-5),
        "showlegend": False,
        "hoverinfo": "none",
        "textinfo": "label",
        "textposition": "outside"
    }
    runfile='/runtime.log'
    #runfile='kashmir_cloud.png'    
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime1, ctime) = os.stat(runfile)

    tm=time.ctime(mtime1)
    
    
    snt=170+sentiment_score*(-5)
    if snt< 60:
        mood='Woo! Awesome!'
    if snt>=60 and snt <120:
        mood='Good!'
    if snt>=120 and snt <180:
        mood='Okay!'
    if snt>=180 and snt <240:
        mood='Not Good!'   
 
        
    if snt>=240 and snt <300:
        mood='Bad as hell!'
    if snt>=300:
        mood='What the hell!'

    
    layout = {

    'paper_bgcolor':'#EEEEEE',
    'plot_bgcolor':'#EEEEEE',
    'margin':dict(t=0,b=0,l=60,r=0,pad=0,autoexpand=True),
    'autosize':False,
    'width':500,
    'height':300,

        'xaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
            
          
        },
        'yaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'shapes': [
            {
                'type': 'path',
                'path': 'M 0.235 0.5 L 0.24 0.65 L 0.245 0.5 Z',
                'fillcolor': 'rgba(0, 0, 0, 0.9)',
                'line': {
                    'width': 1
                },
                'xref': 'paper',
                'yref': 'paper'
            }
        ],
        'annotations': [
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.23,
                'y': 0.45,
                'text': mood,
                'showarrow': False
                
            }
        ],
                'annotations': [
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.63,
                'y': 0,
                'text': 'Last Updated :'+tm,
                'showarrow': False,
                'font':dict(family='Helvetica',size=8)
                
            }
        ],
        
    }
    
    # we don't want the boundary now
    base_chart['marker']['line']['width'] = 0
    
    fig = {"data": [base_chart],
           "layout": layout}
    py.sign_in('xxx', 'xxx')
    py.plot(fig, filename='gauge-meter-chart')


ks='//twitter_kashmir.txt'


ks=pd.read_table(ks) 
ks=ks.tail(500)

x1=[]
y1=[]
       
x,y=0,0
for o in ks.index:
    x+=1
    if 'pos' in ks.ix[o]['pos']:
        y+=1
    elif 'neg' in ks.ix[o]['pos']:
        y-=1
        
    x1.append(x)
    y1.append(y)  

py.sign_in('xxx', 'xxx')
sentiment_gauge(y1[-1])    
    
