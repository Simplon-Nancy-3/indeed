from bs4 import BeautifulSoup
import requests
import re, json
import ast

# req = requests.get('https://www.indeed.fr/emplois?q=data scientist&l&advn=1699844263147621&vjk=742278d478124f60')
# with open("html/sample_0.html", 'w') as f:
#     f.write(BeautifulSoup(req.text, features="html.parser").prettify())

def dict_from_str(dict_str):
    kv = dict_str.split(',')
    for i in range(len(kv)):
        k, v = kv[i].split(':')        
        kv[i] = '\'{}\':{}'.format(k, v) 
    return '{%s}'%(','.join(kv))


with open("html/sample_0.html", 'r') as f:
    with open('json/sample_0.json', 'w') as fw:
        fw.write(json.dumps([json.loads(dict_from_str(hit).replace('\'', '"')) for hit in re.findall(r'jobmap\[[0-9]+\]= {([^}]*)}', f.read())]))