import pandas as pd
import os
import time
from datetime import datetime
import re

#read ratio data from location
path='C:\Users\Arslan Qadri\Google Drive\Programs\FinanceData\_KeyStats'

#read sp500 ratio from csv
sp_data=pd.DataFrame.from_csv('SP500.csv')


#function takes in the name of the ration, and finds it from the source code of html pages
#it also appends corresponding the SP500 data to the data
def keystats(folder_path,ratio='Total Debt/Equity (mrq)'):
    folder_files=os.walk(folder_path)               #find folders in folder
    stock_list_dir = [x[0] for x in folder_files]    #first in the list is a list of folderd followed by list of files
    d=[]        #list to hold data
    
    for each_dir in stock_list_dir[1:]:
        file_list=os.listdir(each_dir)
        if len(file_list)>0:                #check file existence
            sp_old=False
            stock_old=False
            for each_file in file_list:     #for every file in sub folder
                try:
                                        
                    date_stamp=datetime.strptime(each_file,'%Y%m%d%H%M%S.html') #filename is in date format to parse it
                    unix_time=time.mktime(date_stamp.timetuple())
                    file_path=each_dir+'\\'+each_file
                    
                    source=open(file_path,'r').read()       #read the html source code
                    source=source.replace('\n','')          #clean data
                    
                    
                    val=source.split(ratio+':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]   #required value in tag

                                    
                    source=source.replace(' ','')           #clean data
                    try:
                        close_val=float(source.split(':</small><big><b>')[1].split('</b></big>')[0])
                    
                    except:
                        #the problem was that inplace of number, it was returning HTML tags 
                        
                        close_val=(source.split(':</small><big><b>')[1].split('</b></big>')[0])                                    
                        close_val=re.search(r'(\d{1,8}\.\d{1,8})',close_val)                       
                        close_val=float(close_val.group(1))                        
                    
                    try:    # from the corresponding data pick up SP500 value
                        spdate=datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                        row=sp_data[(sp_data.index==spdate)]
                        sp500=float(row['Close'])
                                        
                    except:  #if value unavailable for that day, pick vale for day-3
                        
                        spdate=datetime.fromtimestamp(unix_time-259200).strftime('%Y-%m-%d')
                        row=sp_data[(sp_data.index==spdate)]
                        sp500=float(row['Close']) 
                    #append all data to the list  
                        
                    if not sp_old:
                        sp_old=sp500
                    if not stock_old:
                        stock_old=close_val
                    
                    sp_change=0
                    stock_change=0
                    
                    if sp_old != False :
                        sp_change=(sp500-sp_old)/sp_old
                        stock_change=(close_val-stock_old)/stock_old
                    
                    difference=stock_change-sp_change        

                    if difference > 0 :
                        status='Good'
                    else:
                        status='Bad'
                    
                    d.append([date_stamp,unix_time,val,each_dir.split("\\")[-1],
                              close_val,stock_change,
                              sp500,sp_change, difference,status])
                except Exception as e:
                    print str(e)
                    
    return d                    
                    
   

frame=pd.DataFrame(keystats(path),columns=['Date','Unix','Value','Stock',
                                           'Close','StockChange','SP500',
                                           'SPChange','Difference','Status'])
frame.to_csv('TDEquity.csv')
