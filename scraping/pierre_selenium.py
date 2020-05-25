import json, csv
import pandas as pd

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pymongo import MongoClient

import time

QUERY = 'ressources+humaines' #data+scientist #data+analyst #business+intelligence #devellopeur
TABLE = 'jobs_rh'
features = [
    {'name':'title', 
        'xpath':'//div[@data-jk="{}"]/*[@class="title"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: x.click(), 
        'errors':[] },
    {'name':'company', 
        'xpath':'//div[@data-jk="{}"]/*[@class="sjcl"]/div/*[@class="company"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="vjs-desc"]'))), 
        'errors':[]},
    {'name':'rating_mean', 
        'xpath':'//div[@data-jk="{}"]/*[@class="sjcl"]/div/*[@class="ratingsDisplay"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'location', 
        'xpath':'//div[@data-jk="{}"]/*[@class="sjcl"]/*[@class="location accessible-contrast-color-location"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'salary', 
        'xpath':'//div[@data-jk="{}"]/div[@class="salarySnippet salarySnippetDemphasizeholisticSalary"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'summary', 
        'xpath':'//div[@data-jk="{}"]/*[@class="summary"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'sponso', 
        'xpath':'//div[@data-jk="{}"]/*[@class="jobsearch-SerpJobCard-footer"]/div/div/div/*[@class=" sjLabelGray "]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'day_since', 
        'xpath':'//div[@data-jk="{}"]/*[@class="jobsearch-SerpJobCard-footer"]/div/div/div/*[@class="date "]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]},
    {'name':'rating_count', 
        'xpath':'//*[@class="slNoUnderline"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]}, 
    {'name':'contract', 
        'xpath':'//*[@class="jobMetadataHeader-itemWithIcon-icon jobMetadataHeader-itemWithIcon-icon-jobs"]/parent::div', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]}, 
    {'name':'desc', 
        'xpath':'//div[@id="vjs-desc"]', 
        'extract':lambda x: x.text, 
        'callback':lambda x: None, 
        'errors':[]}]

client = MongoClient('mongodb://localhost:27017')
db=client.indeed

def find_feature(key, feature, root=''):
    try:
        target = driver.find_element_by_xpath(feature['xpath'].format(key))
        feature['callback'](target)
        return feature['extract'](target)
    except NoSuchElementException as e:
        print('Error : {} >> {}'.format(feature['name'], feature['xpath']))
        feature['errors'].append(key)
        return None


driver = webdriver.Chrome(ChromeDriverManager().install())
driver.set_window_size(1920, 1080)
wait = WebDriverWait(driver, 60*5)

c= 0
for i in range(0, 2000, 10):
    driver.delete_all_cookies()
    driver.get('https://www.indeed.fr/emplois?q={}&start={}'.format(QUERY, i))
    for elem in driver.find_elements_by_class_name('jobsearch-SerpJobCard'):
        c+=1
        job = {'_id':elem.get_attribute('data-jk')}
        job_exist = db[TABLE].find_one(job)
        if not job_exist:
            job['query'] = [QUERY]
            for feature in features:
                job.update({feature['name']:find_feature(job['_id'], feature, root='//div[@data-jk="{}"]/'.format(job['_id']))})
            print(job['company'])
            db[TABLE].insert_one(job)
        else:
            db[TABLE].update_one(job, {"$set":{"query":list(set(job_exist['query']+[QUERY]))}})
    for feature in features:
        print('{} - {}/{} errors.'.format(feature['name'], len(feature['errors']), c))
    print('{}/1000'.format(i))
with open('errors.json', 'w') as f:
    f.write(json.dumps(features, default=lambda o: '<not serializable>'))

client.close()
