import sklearn
import numpy as np
from sklearn import svm, preprocessing
from matplotlib import style
import pandas as pd
import random
import warnings



FEATURES = ['DE Ratio', 'Trailing P/E', 'Price/Sales', 'Price/Book', 'Profit Margin', 'Operating Margin', 'Return on Assets', 'Return on Equity', 'Revenue Per Share', 'Market Cap', 'Enterprise Value', 'Forward P/E', 'PEG Ratio', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA', 'Revenue', 'Gross Profit', 'EBITDA', 'Net Income Avl to Common ', 'Diluted EPS', 'Earnings Growth', 'Revenue Growth', 'Total Cash', 'Total Cash Per Share', 'Total Debt', 'Current Ratio', 'Book Value Per Share', 'Cash Flow', 'Beta', 'Held by Insiders', 'Held by Institutions', 'Shares Short (as of', 'Short Ratio', 'Short % of Float', 'Shares Short (prior ']

style.use('ggplot')

df2=pd.DataFrame.from_csv('key_stats.csv')

df2=df2.reindex(np.random.permutation(df2.index))

#df2=df2[:]
df=(df2[FEATURES].values)

df=preprocessing.scale(df)

status=(df2['Status'].replace('underperform',0).replace('outperform',1).values.tolist())

t=500
 


clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(df[:-t],status[:-t])

cc=0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for x in range(1,t+1):
        if clf.predict(df[-x])[0]==status[-x]:

            cc+=1
print cc    
print 'Accuracy:', (cc/float(t)*100 ) 


      





