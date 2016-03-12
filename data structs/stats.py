import pandas as pd
import urllib as url
import matplotlib.pyplot as plt
import matplotlib.style as s

s.use('fivethirtyeight')

urlo='https://www.quandl.com/api/v3/datasets/NSE/JKBANK.csv?auth_token=pi2ReHjeHKdEH38shGDR'

def pick(string):
    read=url.urlopen(string)
    df=pd.read_csv(read)
    df.to_pickle('jkb.pickle')
    return df

df=pick(urlo)
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)       

#slic=df.ix[1:10,'Date':'Close']
#slic.sort('Open',inplace=True)
#
#

#
#df.sort('Date',inplace=True)
#

#
#df2=df.resample('1M',how='mean')
#
#df['Close'].plot()
#df2['Close'].plot()


df2['AOpen']=df['Open'].resample('1M',how='mean')

df2['AOpen'].plot()

df3=df2['AOpen'].resample('1M',how='ohlc')

if df3.values.isnull()
plt.show()



