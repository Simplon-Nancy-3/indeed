# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:31:29 2020

@author: Vincent
"""

# =============================================================================
# Récupération du csv avec plages de salaire
# Ajout de 3 colonnes avec salaire min / max / moyen mensuel -- en cours
# =============================================================================

import numpy as np
import pandas as pd
import html2text
import math

## Chargement du cs de Pierre
path = 'C:/Users/Utilisateur/Google Drive/Simplon/Formation/indeed.projet/indeed-pierre/'
#df = pd.read_csv(path+'indeed.csv', keep_default_na=False)  ## Les nan sont convertis en null
df = pd.read_csv(path+'indeed.csv')  ## conserver les nan pour intégration dans le scrapping


# Suppression des nan pour récupérer les différentes valeurs possibles des salaires proposés
# list = pd.DataFrame(set(df["salary"]),columns =['Salaire']).dropna()


# =============================================================================
# Conversion HTML -> texte puis calcul du salaire mensuel (plage / fixe)
# =============================================================================
import html2text
to_text = html2text.HTML2Text()


# Calcul des salaires mini / maxi et mensuel
#
def mensuel(salaire) :
    ## passage de HTML en texte puis suppression accolades et sauts de ligne
    msalaire=mini=maxi=0
    salaire = to_text.handle(salaire).replace("{", "").replace("}", "").replace("\n\n", "")
    if salaire.find('-') > 0 :  # Plage de salaire
        pos1 = salaire.find('€')
        pos2 = salaire.find('€', pos1)
        mini = int(salaire[:pos1-1].replace(" ", ""))
        maxi = int(salaire[pos1+4:pos1+3+pos2].replace(" ", ""))
        msalaire = (mini + maxi)/2
    else :
        pos1 = salaire.find('€')
        msalaire = mini = maxi = int(salaire[:pos1-1].replace(" ", ""))

    periode = salaire[salaire.find('par'):]

    if periode == 'par an':
        msalaire = msalaire / 12
        mini = mini / 12
        maxi = maxi / 12
    elif periode == 'par heure':
        msalaire = msalaire * 8 * 22
        mini = mini * 8 * 22
        maxi = maxi * 8 * 22
    elif periode == 'par jour' :
        msalaire = msalaire * 22        
        mini = mini * 22
        maxi = maxi * 22
   
    return(mini, maxi, msalaire)

# =============================================================================
# Ajout des 3 colonnes pour chaque enrregistrement
# Si nan => on laisse en planc
# =============================================================================
mini=[]
maxi=[]
moyen=[]
for num in range(len(df)) :
    offre = df['salary'][num]
    #if offre != 'null' :  ## utilisation avec import csv avec nettoyage nan en null
    if type(offre) == type(str()):
        print(offre, mensuel(offre)[0], mensuel(offre)[1], mensuel(offre)[2])
        mini.append(mensuel(offre)[0])
        maxi.append(mensuel(offre)[1])
        moyen.append(mensuel(offre)[2])
    else:
        mini.append('')
        maxi.append('')
        moyen.append('')

df['mini']=pd.DataFrame(mini,columns =['mini'])
df['maxi']=pd.DataFrame(maxi,columns =['maxi'])
df['moyen']=pd.DataFrame(moyen,columns =['moyen'])


# =============================================================================
# Export du dataframe complété
# =============================================================================
df.to_csv(r''+path+'indeed-sal.csv')




