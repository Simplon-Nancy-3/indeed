from bs4 import BeautifulSoup
import requests
import re, json
import ast

def extract_kv(str_):
    kv = str_.split('\',')
    for i in range(len(kv)):
        k, v = kv[i].split(':') 
        kv[i] = (k, v[1:-1])
    return kv

def dict_from_str(str_, dict_):
    kv = extract_kv(str_)
    dict_.update({kv[0][1]:{k:v for k, v in kv[1:]}})
    return dict_

# req = requests.get('https://www.indeed.fr/emplois?q=data+scientist&start=0')
# with open("html/sample_0.html", 'w') as f:
#     f.write(BeautifulSoup(req.text, features="html.parser").prettify())

res = {}
with open("html/sample_0.html", 'r') as f:
    for hit in re.findall(r'jobmap\[[0-9]+\]= {([^}]*)}', f.read()):
        dict_from_str(hit, res)

with open('json/sample_0.json', 'w') as fw:
    fw.write(json.dumps(res))