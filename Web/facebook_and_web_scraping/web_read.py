#both libs r same
import urllib
import urllib2


#url and parameter adding
url='http://google.com/search?'
search={'q':'arslan qadri ka jagrata'}


#encoding, removing spaces and replacing them with +
dat=urllib.urlencode(search)

#dat=dat.encode('utf-8')
url=url+dat

#add headers to disguise as humans
headers={}
headers['User-Agent']="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36"

#adds headers, this statemenet can be skipped if only URL is passed
req=urllib2.Request(url,headers=headers)

#open url, if only URL is passed, use this
rq=urllib2.urlopen(req)


d=rq.read()
print d