from bs4 import BeautifulSoup
import lxml
import requests



url="http://poj.org/problem?id=1000"
headers={
'Host': 'poj.org',
'Connection': 'keep-alive',
'Pragma': 'no-cache',
'Cache-Control': 'no-cache',
'Upgrade-Insecure-Requests': '1',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
'Accept-Encoding': 'gzip, deflate',
'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,en-US;q=0.6',
'Cookie':'pgv_pvi=8264142848; pgv_si=s6157391872; JSESSIONID=D7807852CBBF54192F75C3A7D884FFFA',
}
strhtml = requests.get(url,headers=headers)
text = strhtml.text
soup = BeautifulSoup(text, 'lxml')
#text = request.urlopen(url).read().decode()
print(soup)
