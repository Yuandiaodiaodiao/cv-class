import requests

headers={
"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
,"Accept-Encoding": "gzip, deflate"
,"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,en-US;q=0.6"
,"Cache-Control": "no-cache"
,'Connection': "keep-alive"
,'Host': 'poj.org'
,'Pragma': 'no-cache'
,'Upgrade-Insecure-Requests': '1'
,'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
}

res=requests.get("http://poj.org/problemstatus?problem_id=1006",headers=headers)
with open('ans.html','w')as f:
    f.write(res.text)