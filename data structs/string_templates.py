from string import Template 


#using templates for printing string
def main():
    c=[]
    
    dict0={'name':'ufi','age':25}
    dict1={'name':'arslan','age':27}
    c.append(dict0)    
    c.append(dict1)    
    t=Template('$name ki age $age hai')
  
    for x in c:
        print t.substitute(x)
    
      