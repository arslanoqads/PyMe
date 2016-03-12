# convert string to number
#accepts values such as  '4,5353,453'   '54.34.345.2'
#brackets must not be present, if there are, either replace them or pass string as str[1:-1]

import numpy as np

def StrToInt(string,sep):
    temp=np.fromstring(string, dtype=int, sep=sep)
    return list(temp)