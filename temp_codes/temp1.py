#<a analytics-link="river" class="chorus-optimize" data-chorus-optimize-id="10794447_*" 
#data-chorus-optimize-content-uid="10794447" data-chorus-optimize-module="*" 
#data-native-ad-id="headline"
# href="http://www.theverge.com/2016/2/17/11030406/neverware-google-
# chromebook-chromium-os-education-microsoft">Microsoft’s dead class
# room PCs are finding new life as speedy Chromebooks</a>
 
 
 
import time
from bs4 import BeautifulSoup
import urllib
import urllib2

      
#from the parsed xml, find all 'description' tags. To get juse the text use .text as below
#for item in xm.find_all('description')[1:]:
#    print item.text

#
##this code searches links, n then iterates through those links.
#for item in xm.find_all('link')[1:]:
#    
#    
#    urlo=item.text
#    #read 2
#    #opens each link and search for text in the<p> tags
#    d=urllib.urlopen(urlo).read()
#    
#    xd=BeautifulSoup(d)
#    for pk in xd.findAll('p'):
#        print pk.text