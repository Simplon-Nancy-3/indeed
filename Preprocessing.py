import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import math
import re

def getdata(urlfiles):
    data = pd.read_csv(urlfiles)
    data.pop('Unnamed: 0')
    data = data.drop_duplicates(subset =['Title', 'Company', 'Location', 'Salary', 'Description','TypeContrat'], keep= 'first',inplace=False)
    return data

def MakeDateGreatAgain(df):
    dede = []
    for k,i in df.iterrows() : 
        if i['Date'].lower().replace(' ','') == "publiéeàl'instant" or i['Date'].lower().replace(' ','') == "aujourd'hui":
            d = datetime.today()
            df['Date'][k] = d
        elif i['Date'].lower().replace(' ','') == "ilya30+jours":
            d = datetime.today() - timedelta(days=31)
            df['Date'][k] = d
        else: 
            if len(i['Date'].split()) == 5 :
                if i['Date'].split()[3].isdigit():
                    d = datetime.today() - timedelta(days=int(i['Date'].split()[3]))
                    df['Date'][k] = d
                else : 
                    df.drop(k)
            else :
                dede.append(k)
    return df.drop(dede)

def Makedummiestypecontrat(data):
    doto = pd.DataFrame(columns= ['TypeContrat_CDI','TypeContrat_Temps plein','TypeContrat_Apprentissage','TypeContrat_Contrat pro','TypeContrat_CDD','TypeContrat_Intérim','TypeContrat_Temps partiel','TypeContrat_Stage','TypeContrat_Indépendant','TypeContrat_notfound'])
    for key ,row in data.iterrows():
        doto.loc[key]= 0
        for typecont in row['TypeContrat'].split(','):
            if typecont.strip() == 'Freelance / Indépendant':
                namecol = 'TypeContrat_'+ 'Indépendant'
                doto[namecol][key] =1
            else :
                namecol = 'TypeContrat_'+ typecont.strip()
                doto[namecol][key] =1
                
    data = pd.concat([data, doto],axis=1)
    data.drop(['TypeContrat_notfound'],axis=1,inplace=True)
    return data

def getmeansalary(df):
    for key ,row in df.iterrows():
        if row['Salary'] == "notfound":
            df['Salary'][key] = 'notfound'
        else: 
            non_decimal = re.compile(r'[^\d.]+')
            separatenum = str(row['Salary']).split('-')
            if len(separatenum) == 1 :
                if 'par an' in separatenum[0]:
                    justeprix= non_decimal.sub('', separatenum[0])
                    df['Salary'][key] = int(justeprix)
                elif 'par mois' in separatenum[0]:
                    justeprix= non_decimal.sub('', separatenum[0])
                    df['Salary'][key] = int(justeprix)*12
                elif 'par jour' in separatenum[0]:
                    justeprix= non_decimal.sub('', separatenum[0])
                    df['Salary'][key] = int(justeprix)*22*12
                elif 'par semaine' in separatenum[0]:
                    justeprix= non_decimal.sub('', separatenum[0])
                    df['Salary'][key] = int(justeprix)*4*12
                elif 'par heure' in separatenum[0]:
                    justeprix= non_decimal.sub('', separatenum[0])
                    df['Salary'][key] = int(justeprix)*8*22*12
            else: 
                if 'par an' in separatenum[1]:
                    bastranche= non_decimal.sub('', separatenum[0])
                    hauttranche= non_decimal.sub('', separatenum[1])
                    moyennealannee= np.mean([int(bastranche),int(hauttranche)])
                    df['Salary'][key] = moyennealannee
                elif 'par mois' in separatenum[1]:
                    bastranche= non_decimal.sub('', separatenum[0])
                    hauttranche= non_decimal.sub('', separatenum[1])
                    moyennealannee= np.mean([int(bastranche)*12,int(hauttranche)*12])
                    df['Salary'][key] = moyennealannee
                elif 'par jour' in separatenum[1]:
                    bastranche= non_decimal.sub('', separatenum[0])
                    hauttranche= non_decimal.sub('', separatenum[1])
                    moyennealannee= np.mean([int(bastranche)*22*12,int(hauttranche)*22*12])
                    df['Salary'][key] = moyennealannee
                elif 'par semaine' in separatenum[1]:
                    bastranche= non_decimal.sub('', separatenum[0])
                    hauttranche= non_decimal.sub('', separatenum[1])
                    moyennealannee= np.mean([int(bastranche)*4*12,int(hauttranche)*4*12])
                    df['Salary'][key] = moyennealannee
                elif 'par heure' in separatenum[1]:
                    bastranche= non_decimal.sub('', separatenum[0])
                    hauttranche= non_decimal.sub('', separatenum[1])
                    moyennealannee= np.mean([int(bastranche)*8*22*12,int(hauttranche)*8*22*12])
                    df['Salary'][key] = moyennealannee
                else: 
                    print("Error : Tout n'est pas passé ")
    
    return df
                

def GetcategLocation(df):
    for key, row in df.iterrows():
        if '75' in row['Location'] or '92' in row['Location'] or '94' in row['Location'] or '93' in row['Location'] or '78' in row['Location'] or '95' in row['Location'] or '77' in row['Location'] or '91' in row['Location'] or 'Hauts-de-Seine'in row['Location'] or 'Val-de-Marne' in row['Location'] or 'Île-de-France' in row['Location']  :
            df['Location'][key] = 'Paris'
        elif '69' in row['Location'] or 'Auvergne-Rhône-Alpes' in row['Location'] or 'Occitanie' in row['Location'] or 'Rhône' in row['Location'] or '38' in row['Location'] :
            df['Location'][key] = 'Lyon'
        elif '54' in row['Location'] or "57" in row['Location'] or 'Meurthe-et-Moselle' in row['Location'] or '01'in row['Location'] or 'Meurthe-et-Moselle'in row['Location'] or 'Grand Est' in row['Location']:
            df['Location'][key] = 'Nancy'
        elif '33' in row['Location'] or 'Gironde' in row['Location'] or 'Haute-Garonne' in row['Location'] :
            df['Location'][key] = 'Bordeaux'
        elif '13' in row['Location'] or "Provence-Alpes-Côte d'Azur" in row['Location']:
            df['Location'][key] = 'Marseille'
        elif '31' in row['Location']  or 'Nouvelle-Aquitaine' in row['Location']  :
            df['Location'][key] = 'Toulouse'
        elif '44' in row['Location']  or 'Pays de la Loire' in row['Location'] or 'Loire-Atlantique' in row['Location']:
            df['Location'][key] = 'Nantes'
        else : 
            df['Location'][key] = row['Location']
    return df


def getparsedtitle(df):
    for key, row in df.iterrows() : 
        if 'data scientist' in row['Title'].lower() or 'datascientist' in row['Title'].lower() or 'scientist' in row['Title'].lower():
            df['Title'][key] = 'Data Scientist'
        elif 'machine learning' in row['Title'].lower()  or 'ml' in row['Title'].lower() or 'intelligence artificielle' in row['Title'].lower() or 'artificial' in row['Title'].lower()  or 'ia' in row['Title'].lower(): 
            df['Title'][key] = 'Machine Learning'
        elif 'data analyst' in row['Title'].lower() or 'analyste' in row['Title'].lower() or 'analyst' in row['Title'].lower(): 
            df['Title'][key] = 'Data Analyst'
        elif 'data engineer' in row['Title'].lower() or 'ingénieur data' in row['Title'].lower() or 'data ingénieur' in row['Title'].lower(): 
            df['Title'][key] = 'Data Engineer'
        elif 'big data' in row['Title'].lower() or 'cloud' in row['Title'].lower() : 
            df['Title'][key] = 'Big Data'
        elif 'manager' in row['Title'].lower() : 
            df['Title'][key] = 'Manager'
        elif 'security' in row['Title'].lower() or 'sécurité' in row['Title'].lower() : 
            df['Title'][key] = 'Cyber sécurity'
        elif 'business intelligence' in row['Title'].lower() or 'business' in row['Title'].lower() : 
            df['Title'][key] = 'Business Intelligence'
        elif 'consultant' in row['Title'].lower() : 
            df['Title'][key] = 'Consultant'
        elif 'data' in row['Title'].lower()or 'business developper' in row['Title'].lower() : 
            df['Title'][key] = 'Data'
        elif 'ingenieur' in row['Title'].lower() or 'ingénieur' in row['Title'].lower() or 'engineer' in row['Title'].lower(): 
            df['Title'][key] = 'Ingénieur'
        elif 'web' in row['Title'].lower() or 'fullstack'in row['Title'].lower().replace('-','').replace(' ','') or 'intégrateur' in row['Title'].lower() : 
            df['Title'][key] = 'Dev Web'
        elif 'web' in row['Title'].lower() or 'frontend'in row['Title'].lower().replace('-','').replace(' ','') or 'javascript'in row['Title'].lower() : 
            df['Title'][key] = 'Dev Front end'
        elif 'php'in row['Title'].lower() or  'backend'in row['Title'].lower().replace('-','').replace(' ','')  : 
            df['Title'][key] = 'Dev Back end'
        elif 'scrum' in row['Title'].lower(): 
            df['Title'][key] = 'Scrum'
        elif 'bio' in row['Title'].lower(): 
            df['Title'][key] = 'Bio-Informatique'
        elif 'ux' in row['Title'].lower() or 'user' in row['Title'].lower() : 
            df['Title'][key] = 'Experience Utilisateur'
        elif 'system' in row['Title'].lower() or 'server' in row['Title'].lower() or 'admin' in row['Title'].lower()  : 
            df['Title'][key] = 'Admin reseau'
        elif 'développeur' in row['Title'].lower() or 'developpeur' in row['Title'].lower() or 'développeu' in row['Title'].lower() or 'developer' in row['Title'].lower() or 'dev' in row['Title'].lower() or 'SAP' in row['Title']: 
            df['Title'][key] = 'developpeur'
        elif 'marketing' in row['Title'].lower(): 
            df['Title'][key] = 'Marketing'
        elif 'lead' in row['Title'].lower() or 'gestion' in row['Title'].lower() or 'chef de projet' in row['Title'].lower() or 'directeur' in row['Title'].lower() or 'responsable' in row['Title'].lower() or 'chief' in row['Title'].lower() : 
            df['Title'][key] = 'Direction'
        elif 'tresorier' in row['Title'].lower() or 'trésorier' in row['Title'].lower() : 
            df['Title'][key] = 'trésorier'
        else : 
            df['Title'][key] = 'autres'
    return df

def getpreprocessdata(df):
    # Oui tout pouerrais tourner sur la meme boucle, mais c'est pas si simple ! 
    df = MakeDateGreatAgain(df)
    df = Makedummiestypecontrat(df)
    df= getmeansalary(df)
    df = GetcategLocation(df)
    df = getparsedtitle(df)
    return df

def getotherDFone():
    doto= pd.read_csv('data/3500rows+jobville.csv')
    doto = doto.drop(['Job','Ville','Unnamed: 0'],axis=1)
    doto = doto.drop_duplicates(subset =['Title', 'Company', 'Location', 'Salary', 'Description','TypeContrat'], keep= 'first',inplace=False)
    doto = getpreprocessdata(doto)
    return doto

def getotherDFtwo():
    dete= pd.read_csv('data/1185rows.csv')
    dete = dete.drop('Unnamed: 0',axis= 1)
    dete = getpreprocessdata(dete)
    return dete

def getotherDFthree():
    dyty= pd.read_csv('data/1030entree.csv')
    dyty = dyty.drop('Unnamed: 0',axis=1)
    dyty = getpreprocessdata(dyty)
    return dyty


def concatotherDF():
    doto = getotherDFone()
    dete = getotherDFtwo()
    dyty = getotherDFthree()
    diti = pd.concat([doto,dete,dyty],axis=0)
    diti = diti.drop_duplicates(subset =['Title', 'Company', 'Location', 'Salary', 'Description','TypeContrat'], keep= 'first',inplace=False)
    return diti

def getDFcleanedFulled():
    data = getdata('test.csv')
    data= getpreprocessdata(data)
    diti = concatotherDF()
    dutu = pd.concat([data,diti],axis=0)
    dutu = dutu.drop_duplicates(subset =['Title', 'Company', 'Location', 'Salary', 'Description','TypeContrat'], keep= 'first',inplace=False)
    return dutu


def Save_to_mongo_CSV(df, collectionname):
    from Connection import insertDFtoMongo
    insertDFtoMongo(df,collectionname)
    df.to_csv(r'C:/Users/Utilisateur/Videos/Python/semaine18distance9projet3/indeed/data/' + collectionname + '.csv' )