import time
from bs4 import BeautifulSoup
import urllib
import urllib2


url='http://www.greaterkashmir.com/feed.aspx?cat_id=2'

#read 1
req=urllib.urlopen(url)

#parse xml
xm=BeautifulSoup(req,'xml')

#from the parsed xml, find all 'description' tags. To get juse the text use .text as below
#for item in xm.find_all('description')[1:]:
#    print item.text


#this code searches links, n then iterates through those links.
for item in xm.find_all('link')[1:]:
    urlo=item.text
    #read 2
    #opens each link and search for text in the<p> tags
    d=urllib.urlopen(urlo).read()
    
    xd=BeautifulSoup(d)
    for pk in xd.findAll('p'):
        print pk.text
    #print url
    
#print xm