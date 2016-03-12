
def trim(frame,column,start,end):    
    i=0
    while i<len(frame):
        frame[column][i]=str(frame[column][i])[start:end]
        i=i+1
        