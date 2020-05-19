import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import math
import re

def getdata(urlfiles):
    data = pd.read_csv(urlfiles)
    data.pop('Unnamed: 0')
    data = data.drop_duplicates(subset=['Link'], keep='first')
    return data

def MakeDateGreatAgain(df):
    indexadel = []
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
                indexadel.append(k)
    return df.drop(indexadel)

def Makedummiestypecontrat(data):
    doto = pd.DataFrame(columns= ['TypeContrat_CDI','TypeContrat_Temps plein','TypeContrat_Apprentissage','TypeContrat_Contrat pro','TypeContrat_CDD','TypeContrat_Intérim','TypeContrat_Temps partiel','TypeContrat_Stage','TypeContrat_Indépendant','TypeContrat_notfound'])
    for key ,row in data.iterrows():
        for typecont in row['TypeContrat'].split(','):
            if typecont.strip() == 'Freelance / Indépendant':
                namecol = 'TypeContrat_'+ 'Indépendant'
                doto.loc[key]= 0
                doto[namecol][key] =1
            else :
                namecol = 'TypeContrat_'+ typecont.strip()
                doto.loc[key]= 0
                doto[namecol][key] =1
                
    data = pd.concat([data, doto],axis=1)
    data = data.drop('TypeContrat',axis=1)
    return data

def getmeansalary(df):
    for key ,row in df.iterrows():
        if row['Salary'] == "notfound":
            df['Salary'][key] = 'notfound'
        else: 
            non_decimal = re.compile(r'[^\d.]+')
            separatenum = row['Salary'].split('-')
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

def getpreprocessdata(urlfiles):
    df = getdata(urlfiles)
    df = MakeDateGreatAgain(df)
    df = Makedummiestypecontrat(df)
    df= getmeansalary(df)
    return df
