import pandas as pd
import os
import time
from datetime import datetime

path='C:\Users\Arslan Qadri\Google Drive\Programs\Quran_corpus'

def ayahCollector(folder_path):
    folder_files=os.walk(folder_path)
    list_dir = [x[0] for x in folder_files]
    
    quran=[]
    
    for each_dir in list_dir:
        
        file_list=os.listdir(each_dir)
        if len(file_list)>0:
            for each_file in file_list:
                file_name='%s:' % str(each_file)[1:7]
                load_file_name=path+'\\'+each_file
                content=open(load_file_name).read()
                ayah=file_name+content
                quran.append(ayah)
    with open('C:\Users\Arslan Qadri\Google Drive\Programs\Quran_corpus\quran.txt','w') as f:
        for i in quran:        
            f.write(i+'\n')                

ayahCollector(path)       



    with open('log.txt','a') as f:       
            f.write(e)  