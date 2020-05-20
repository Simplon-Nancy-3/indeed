from bs4 import BeautifulSoup
import requests
import re, json
import ast

from pymongo import MongoClient

# data+scientist, data+scientist, developpeur
QUERY = 'business+intelligence'

client = MongoClient('mongodb://localhost:27017')
db=client.indeed

def get_jobs(q, start):
    url = 'https://www.indeed.fr/emplois?q={}&start={}'.format(q, start)
    req = requests.get(url)
    return url, re.findall(r'jobmap\[[0-9]+\]= {([^}]*)}', req.text)
def get_job():
    return BeautifulSoup(requests.get(job['url']).text, features="html.parser")

def parse_js(_str):
    return {i.split(':')[0]:(':'.join(i.split(':')[1:])[1:]) for i in _str.split('\',')}

def job_exist(_id):
    return db['jobs_test'].count_documents({'_id':_id}) > 0
def insert_job(job):
    job['_id'] = job['jk']
    del job['jk']
    db['jobs_test'].insert_one(job)


def parse_url(job):
    job['url'] = 'https://www.indeed.fr/voir-emploi?jk={}'.format(job['jk'])
def parse_meta(job, soup):
    job['salary'] = None
    for div in soup.find_all('div', attrs={'class': ['jobsearch-InlineCompanyRating', 'icl-u-xs-mt--xs', 'jobsearch-DesktopStickyContainer-companyrating']}):
        job.update({'%s'%div.find('div')['class'][1].split('--')[-1]:div.text})
def parse_desc(job, soup):
    desc = soup.find('div', attrs={'id':"jobDescriptionText"})
    job.update({'desc': desc.text if desc != None else ''})
def parse_footer(job, soup):
    footer = soup.find('div', attrs={'class':'jobsearch-JobMetadataFooter'})
    job.update({'footer':footer.text if footer != None else None})

for i in range(0, 10000, 10):
    url, jobs = get_jobs(QUERY, i)
    for j in range(0, len(jobs)):
        job = parse_js(jobs[j])
        if not job_exist(job['jk']):
            parse_url(job)
            soup = get_job()
            parse_meta(job,soup)
            parse_desc(job,soup)
            parse_footer(job,soup)
            insert_job(job)
            print('{}/{} - {}'.format(j, len(jobs), job['url']))
    print('{}/{} - {}'.format(i+10, 10000, url))

client.close()


