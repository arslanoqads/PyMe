import pandas as pd
import matplotlib.pyplot as plt
import urllib
import pickle
import time


#Read csv
#Change index
#Plot data
#Write to csv
#CSV read options
df=pd.read_csv('FBI.csv')
df.set_index('Year', inplace=True)
#df['Robbery'].plot()
df['Robbery'].to_csv('FBI2')
df2=pd.read_csv('FBI2',index_col=0,names=['Year','Chori'])
#print(df2)
#plt.show()


#create HDF store
#store df
#store df2
#close store
#read stored HDF
#print
store=pd.HDFStore('hdf.h5')
store.put('d1',df,format='table',data_columns=True)
store.put('d2',df2,format='table',data_columns=True)
store.close()
hdf=pd.read_hdf('hdf.h5','d2')
#print(hdf)


#save to json
#read from json
#print json
#
#open a json web location
#read json
#convert to data frame
df.to_json('file_json.json')
df3=pd.read_json('file_json.json')
#print(df3.head())

df4_json=urllib.urlopen('https://api.github.com/users/mralexgray/repos').read()
df4=pd.read_json(df4_json)
#print(df4.head())


#note start time
#read to pickle file
#read from pickle file
#print time taken
start=time.time()
df4.to_pickle('pk.pickle')
df4=pd.read_pickle('pk.pickle')
print(time.time()-start)


#Using pickle package - note start time
#open pickle file 
#write to file
#close
#open pickle file
#load file
#print time taken
start=time.time()
out=open('pk2.pickle','wb')
pickle.dump(df4,out)
out.close()
inp=open('pk2.pickle','rb')
df5=pickle.load(inp)
print(time.time()-start)

#print(df5)




