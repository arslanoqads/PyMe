import pandas as pd


#data frames set
frame1=pd.DataFrame({'c1':[4,3,6,4,3,5],
        'c2':[7,4,5,2,3,4],
        'c3':[7,8,6,8,6,7]},
        index=[1,2,3,4,5,6])
        
frame2=pd.DataFrame({ 'c1':[4,3,6,4,3,5],
        'c2':[7,4,5,2,3,4],
        'c3':[7,8,6,8,6,7]},
        index=[7,8,9,10,11,12])

frame3=pd.DataFrame({ 'c1':[4,3,6,4,3,5],
        'c2':[7,4,5,2,3,4],
        'c3':[7,8,6,8,6,7]},
        index=[13,14,15,16,17,18])
        

#data frame set 2
fram1=pd.DataFrame({'c1':[1,1,1],
        'c2':[1,1,1],
        'c3':[1,1,1]},
        index=[1,2,3])
        
fram2=pd.DataFrame({ 'c1':[2,2,2],
        'c2':[2,2,2],
        'c4':[2,2,2]},
        index=[4,5,6])

fram3=pd.DataFrame({ 'c1':[3,3,3],
        'c2':[3,3,3],
        'c3':[1,1,1]},
        index=[1,2,3])
        
#concatenate data frames       
frame=pd.concat([frame1,frame2,frame3])        
        
#Concatination based on axis : unique column 1. 
#All column 1 corresponding columns are allogned        
df=pd.concat([fram1,fram2,fram3],axis=1)     

#merging
merged2=pd.merge(frame1,frame2,on='c1',how='inner') 
merged=pd.merge(fram1,fram2,on='c1',how='left') 
print(merged)


f1=fram1.append(fram2,ignore_index=True)

fra1=pd.DataFrame({'c11':[4,3,6,4,3,5],
        'c12':[7,4,5,2,3,4],
        'c13':[7,8,6,8,6,7]},
        index=[1,2,3,4,5,6])
        
fra2=pd.DataFrame({ 'c1':[4,3,6,4,3,5],
        'c2':[7,4,5,2,3,4],
        'c3':[7,8,6,8,6,7]},
        index=[1,3,9,10,1,2])

#join
j=fra1.join(fra2,how='inner')