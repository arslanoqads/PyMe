import pandas as pd
import glob
import os
import column_trim as ct

filelist = []
frame=pd.DataFrame()

def trim(frame,column,start,end):    
    i=0
    while i<len(frame):
        frame[column][i]=frame[column][i][start:end]
        print frame[column][i]
        i=i+1
    
j=0
#concatename all .csv files from a folder
os.chdir("C:\Users\Arslan Qadri\Downloads\earthquake")
for counter, files in enumerate(glob.glob("*.csv")):
    df=pd.DataFrame(pd.read_csv(files))
    ct.trim(df,'time',0,19)
    frame=frame.append(df)
    j=j+1
    print 'iter:',j

frame=frame.head()

trim(frame,'time',0,19)

