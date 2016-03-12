import cx_Oracle
from sqlalchemy import create_engine
import pandas as pd
import sqlalchemy



path='C:\Users\Arslan Qadri\Google Drive\Sem 2\Stats Learning\project\\'


#train=pd.read_csv(path+'train.csv')



att=pd.read_csv(path+'attributes.csv')

#desc=pd.read_csv(path+'product_descriptions.csv')

st=att

s=st['product_uid']

sto=[int(str(i)[:-2]) for i in s if len(str(i))>4]
stoc=set(sto)

pdo={}
for i in stoc:
    
    d={}
    name=list(st[st['product_uid']==i]['name'])
    val=list(st[st['product_uid']==i]['value'])
    for k,j in enumerate(name):

        d[j]=list(st[st['product_uid']==i]['value'])[k]
        
    pdo[i]=d    
    

"""
#create a connection string is  : Username/password@127.0.0.1/DB_name
def oracle_con():
    con = cx_Oracle.connect('SYSTEM/arslan@127.0.0.1/XE')
    print con.version
    return con.cursor()
        
cur=oracle_con()


#load data from DB
d=[]
cur.execute('select * from item')
for result in cur:
    d.append(result)

    print result
cur.close()
  

#lookup using bind variables
cur.prepare('select * from item where price=:id')
cur.execute(None, {'id': 231})
for result in cur:
    print result



#insert into DB from pandas DF
c=pd.DataFrame(d,columns=['Item','Class','Dept'])
engine = create_engine('oracle+cx_oracle://SYSTEM:arslan@XE',coerce_to_unicode=True)
c.to_sql('chunky', engine, if_exists='append',dtype={'value': sqlalchemy.types.NUMERIC} )


"""
