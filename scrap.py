from bs4 import BeautifulSoup
import requests

req = requests.get('https://www.indeed.fr/emplois?q=data scientist&l&advn=1699844263147621&vjk=742278d478124f60')
with open("html/sample_0.html", 'w') as f:
    f.write(BeautifulSoup(req.text, features="html.parser").prettify())
