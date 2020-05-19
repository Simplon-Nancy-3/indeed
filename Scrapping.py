import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup
import requests
def scrapPage(urldelapageascrap,job,ville):
    try : 
        req = requests.get(urldelapageascrap)
    except:
        return pd.DataFrame(columns=['Title','Company','Location','TypeContrat','Salary','Description','Date','Link','Job','Ville'])
    soup = BeautifulSoup( req.text , 'lxml')
    url = []
    data = pd.DataFrame(columns=['Title','Company','Location','TypeContrat','Salary','Description','Date','Link','Job','Ville'])
    for jobs in soup.find_all('div', {'class': 'jobsearch-SerpJobCard'}):
        url.append(jobs.find('h2',{'class':'title'}).find('a')['href'])
    for link in url : 
        req = requests.get('https://www.indeed.fr'+link)
        soup = BeautifulSoup( req.text , 'lxml')
        title = soup.find('h3',{'class':'jobsearch-JobInfoHeader-title'})
        if title :
            title = title.text
        else : 
            title = 'notfound'
        company = soup.find('div',{'class':'jobsearch-InlineCompanyRating'}).findChildren()
        if len(company) >= 1  :
            company = company[0].text
        else : 
            company = 'notfound'
        location = soup.find('div',{'class':'icl-IconFunctional icl-IconFunctional--location icl-IconFunctional--md'})
        if location : 
            location = location.find_parent().text
        else : 
            location = 'notfound'
        contrattype = soup.find('div',{'class':'icl-IconFunctional icl-IconFunctional--jobs icl-IconFunctional--md'})
        if contrattype : 
            contrattype = contrattype.find_parent().text
        else : 
            contrattype = 'notfound'
        salary = soup.find('div',{'class':'icl-IconFunctional icl-IconFunctional--salary icl-IconFunctional--md'})
        if salary : 
            salary = salary.find_parent().text
        else : 
            salary = 'notfound'
        date = soup.find('div',{'class':'jobsearch-JobMetadataFooter'})
        if date : 
            date = date.text.split('- ')[1]
        else : 
            date = 'notfound'
        description = soup.find('div',{'id':'jobDescriptionText'})
        if description : 
            description = description.text
        else : 
            description = 'notfound'
        makeappend = pd.DataFrame({'Title':[title],'Company':[company],'Location':[location],'TypeContrat':[contrattype],'Salary':[salary],'Description':[description],'Date':[date],'Link':[link],'Job':[job], 'Ville': [ville]})
        data = pd.concat([data, makeappend], axis=0, sort=False,ignore_index=True)
    return data

def pagination(url,job,ville) : 
    data = pd.DataFrame(columns=['Title','Company','Location','TypeContrat','Salary','Description','Date','Link','Job','Ville'])
    for i in range(0,154,14):
        urlpagine =  url + '&start='+str(i)
        doto = scrapPage(urlpagine,job,ville)
        data = pd.concat([data, doto], axis=0, sort=False,ignore_index=True)
    return data

def makejobetville(jobs,villes):
    data = pd.DataFrame(columns=['Title','Company','Location','TypeContrat','Salary','Description','Date','Link','Job','Ville'])
    for job in jobs :
        for ville in villes: 
            url = 'https://www.indeed.fr/jobs?q=' + job + '&l=' + ville
            doto = pagination(url,job,ville)
            data = pd.concat([data, doto], axis=0, sort=False,ignore_index=True)
    return data



#data = scrapPage("https://www.indeed.fr/jobs?q=dev&l=Paris")
#pagination("https://www.indeed.fr/jobs?q=dev&l=Paris")
data = makejobetville(['Data Scientist','data engineer','data analyst','Business Intelligence'], ['Paris','Bordeaux','Lyon','Nancy','Toulouse','Nantes','Marseille'])
#data = makejobetville(['Data Scientist'], ['Lyon'])
data.to_csv(r'C:\Users\Utilisateur\Videos\Python\semaine18distance9projet3\indeed\test.csv')

print(data)