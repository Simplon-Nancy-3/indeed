from bs4 import BeautifulSoup
import requests
import re, json
import ast

req = requests.get('https://www.indeed.fr/emplois?q=data+scientist&start=0')
for hit in re.findall(r'jobmap\[[0-9]+\]= {jk:\'([a-z0-9]+)\'', req.text):
    job_url = 'https://www.indeed.fr/voir-emploi?jk={}'.format(hit)
    soup = BeautifulSoup(requests.get(job_url).text, features="html.parser")
    job_title = soup.find('div', attrs={'class':'jobsearch-JobInfoHeader-title-container'})
    print({'title':job_title.text})