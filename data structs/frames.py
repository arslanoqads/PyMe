import pandas as pd
from matplotlib import style
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
import urllib as url


style.use('fivethirtyeight')



frame={ 'c1':[4,3,6,4,3,5],
        'c2':[7,4,5,2,3,4],
        'c3':[7,8,6,8,6,7],
        'c4':[4,5,6,4,3,4],
        'c5':[1,2,3,4,4,2],
        'c6':[2,1,3,3,2,1],
        'n':['AR','SA','LA','NQ','AD','RI']
        }
        
df=pd.DataFrame(frame)        

#print(df.dtypes)
#print(df['c1'][1])

#print(df.head(1))

#print(df.tail(1))

#slicing frames
print(df[:][3:4])

#changing index to n
df2=df.set_index('n')


print(df2[:]['AR':'LA'])



#setting dates
start=datetime.datetime(2015,1,1)
end=datetime.datetime(2015,12,31)

#connecting to yahoo api
att=web.DataReader('gmc','yahoo',start,end)

#printing from api
print(att.head())


#plotting specified columns
att[['High','Low']].plot()

#plotting everything
att.plot()

#list : simple an array
high_list=att['High'].tolist()

#series : table like, index+ column
high_series=att['High']

plt.legend()
plt.show()