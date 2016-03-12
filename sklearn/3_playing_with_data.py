import pandas as pd
import datetime as datetime
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
from collections import Counter


style.use('dark_background')

#load data from csv created from the other program
df2=pd.read_csv('TDEquity.csv')

#create a list of stocks with corresponding values available
all_stocks=list(df2['Stock'])
dist=dict(Counter(all_stocks))

#select stocks with more than 30 values each
top_stocks=[i for i,v in dist.items() if v>30][:35]


"""
The date format did not work as expected with Pandas' autmatic
date manipulation. Had to convert the date column into pandas'
date_time format explicitly.
"""


for stock in top_stocks:
    plot_df=df2[df2['Stock']==stock]        # match stock from the created list
    plot_df['Date']=pd.to_datetime(plot_df['Date'])  # make sure the date format is correct
    plot_df=plot_df.set_index(['Date'])  
    
    if plot_df['Status'][-1]=='Bad':
        color='r'
    else:
        color='g'
    
    plot_df['Difference'].plot(label=stock,color=color)
    plt.legend()
    

plt.show()        