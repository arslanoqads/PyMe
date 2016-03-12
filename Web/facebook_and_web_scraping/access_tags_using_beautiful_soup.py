import urllib2, os, sys
from bs4 import BeautifulSoup

#############################################################################
# Author     : Mohamed, Qadri
# Assignment : 3
# Description: The algorithm, autmoatically chooses the BIG STORIES OF THE 
#              DAY from VERGE, and downloads and as many of them as the user 
#              requires. This is decided by the pagesToGet variable.
#############################################################################


# INPUT VARIABLES
url='http://theverge.com/'
headers={'User-Agent':'Mozilla/5.0'}
pagesToGet=3





if not os.path.exists('reviewPages'):os.mkdir('reviewPages')        #create folder if missing

#read the homepage
re=urllib2.Request(url,headers=headers)
#open url, if only URL is passed, use this
req=urllib2.urlopen(re)

i=1 # counter for printed articles
xm=BeautifulSoup(req)    
#for a in xm.find_all('a',attrs={'class':'chorus-optimize','analytics-link':'river'},href=True)[:pagesToGet]:  
for a in xm.find_all('li',class_='m-new-articles__story')[:pagesToGet]:  
    
    try:    
        #The Big stories are present in the <li> tag with class name as above.
        #To find the link inside the <li> tag we use the below syntax to get a 
        #list of values, in which tag#contains the link.
        #HREF is used to print the link
    
        """
        The contents tag returns all the subtags as a list. This can be used
        recursively.
        """
        article_url=a.contents[3]['href']
        
        #read article 
        re=urllib2.Request(article_url,headers=headers)
        req=urllib2.urlopen(re)
        
        #parse article                
        soup = BeautifulSoup(req) 
        
        #find author name
        list = soup.findAll('a', attrs={'class':'author fn'})
        
        #if authors are returned multiple time, take the first
        file=list[0].text 
        
        #replace spaces with _
        file=file.replace(' ','_')
        
        re=urllib2.Request(article_url,headers=headers)
        req=urllib2.urlopen(re)
        
        #write to file, named as author's name
        fwriter=open('reviewPages/'+str(file)+'.html','w')
        fwriter.write(req.read())
        fwriter.close()
        
        #print status of loading
        print 'Article ',i,' loaded.'
        i+=1
        
            
    except Exception as e: # this describes what to do if an exception is thrown
        error_type, error_obj, error_info = sys.exc_info()# get the exception infomration
        print 'ERROR FOR LINK:',article_url #print the link that cause the problem
        print error_type, 'Line:', error_info.tb_lineno #print error info and line that threw the exception
        continue#ignore this page.
     

