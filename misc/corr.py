import pandas as pd
import matplotlib.pyplot as plt
import urllib as url
import matplotlib.style as s

s.use('fivethirtyeight')

def pick(string):
    read=url.urlopen(string)
    df=pd.read_csv(read)
    df.to_pickle('jkb.pickle')
    return df

#population
u1='https://www.quandl.com/api/v3/datasets/PENN/IND_POP.csv?auth_token=pi2ReHjeHKdEH38shGDR'
#exchange
u2='https://www.quandl.com/api/v3/datasets/MOSPI/FRGN_EXCHNG_RSRVS_INDIAN_CURR.csv?auth_token=pi2ReHjeHKdEH38shGDR'
#inflation
u3='https://www.quandl.com/api/v3/datasets/INDIA_LAB/INFLATION.csv?auth_token=pi2ReHjeHKdEH38shGDR'
#trade
u4='https://www.quandl.com/api/v3/datasets/INDIA_COMM/TRADE.csv?auth_token=pi2ReHjeHKdEH38shGDR'


pop=pick(u1)
exh=pick(u2)
inf=pick(u3)
trd=pick(u4)

pop.sort('Date',inplace=True)
inf.sort('Date',inplace=True)
trd.sort('Date',inplace=True)

pop.set_index('Date',inplace=True)
inf.set_index('Date',inplace=True)
trd.set_index('Date',inplace=True)

ti=trd.join(inf,how='inner')

#tip=ti.join(pop,how='inner')


#pid.plot()
#plt.legend()
#plt.show()

print(ti.corr())

ti.plot()