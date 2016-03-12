import pandas as pd
from matplotlib import style
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt



style.use('fivethirtyeight')


#function to use in rolling_apply
def mine(data):
    return max(data)
    
    

#setting dates
start=datetime.datetime(2015,1,1)
end=datetime.datetime(2015,12,31)

#connecting to yahoo api
att=web.DataReader('gmc','yahoo',start,end)

#manipulations
att['Open']=att['Open']/100
att['Check']=att['Close']>att['Open']
att['bool']=att['Check']
#print(att[(att['Close']>att['Open'])])

#print(att.head())

#changing falso to 0 and true to 1
y=[]
for x in range(len(att)):
    if att['Check'][x] == False:
        t=0
        y.append(t)
    else:
        t=1
        y.append(t)
att['bool']=y

#att[['Check']].plot()
#plt.show()

#showing rolling function
att['MA10']=pd.rolling_mean(att['Open'],10)
att['Max10']=pd.rolling_apply(att['Close'],10,mine)
#att[['MA10','Max10']].plot()

#plt.show()

#att['MA10'].fillna(method='ffill')
#att['MA10'].fillna(method='bfill',inplace=True)
#
#att['Max10'].fillna(value=10000,inplace=True)

y=att['MA10'][att['MA10'].isnull()]
print 'Total NaNs are ',len(y)


#print(att.isnull.values.sum())




