# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:49:17 2020

@author: Vincent

Liste des algos de regression linéaire : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

"""



# =============================================================================
# Splittage des offres sur le 5 villes demandées
# =============================================================================

import numpy as np
import pandas as pd
import html2text

## Chargement du cs de Pierre
path = 'C:/Users/Utilisateur/Google Drive/Simplon/Formation/indeed.projet/indeed-pierre/'
#df = pd.read_csv(path+'indeed.csv', keep_default_na=False)  ## Les nan sont convertis en null
df = pd.read_csv(path+'jobs_it_process.csv')  ## conserver les nan pour intégration dans le scrapping

df.columns
df.rename(columns={'location': 'location_dirty','region': 'region_p'}, inplace=True)
df_region = df.dropna(subset=['salary_mean'], axis=0)

'''
# Région Parisienne  = 75 - 77 - 78- 91 - 92 - 93 -94 - 95
# Région Lyonnaise   = 69 - 01- 42 - 38 - 71
# région Bordelaise  = 33 - 17 - 16 - 24 - 47 - 40
# région Nantaise    = 44 - 56 - 35 - 53 - 49 - 85
# Région Toulousaine = 31 - 32 - 82 - 81 - 11 - 9 - 65
'''

# =============================================================================
# Nettoyage des départements manquants : ajout si utile, suppression sinon
# Création des 5 regroupements
# =============================================================================
df.isna().sum()
for i in range(len(df)):
    if pd.isna(df['dep'][i]):
        if df['location_dirty'][i] in ['Paris', 'Île-de-France']:
            df['dep'][i]=75
        elif df['location_dirty'][i] in 'Hauts-de-Seine':
            df['dep'][i]=92
        elif df['location_dirty'][i] in 'Yvelines':
            df['dep'][i]=78
        elif df['location_dirty'][i] == 'Rhône':
            df['dep'][i]=69
        elif df['location_dirty'][i] == 'Loire-Atlantique':
            df['dep'][i]=44
    if df['location_dirty'][i][0:5] == 'Paris':
        df['dep'][i]=75
    if df['location_dirty'][i][0:4] == 'Lyon':
        df['dep'][i]=69
    if df['location_dirty'][i][0:9] == 'Marseille':
        df['dep'][i]=13

df.isna().sum()
df = df.dropna(subset=['dep'], axis=0)
df.isna().sum()

def size(n):
    if n in ['75','77','78','91','92','93','94','95']:
        return 'Paris'
    elif n in ['69', '01', '42', '38', '71']:
        return 'Lyon'
    elif n in ['33', '17', '16', '24', '47', '40']:
        return 'Bordeaux'
    elif n in ['44', '56', '35', '53', '49', '85']:
        return 'Nantes'
    elif n in ['31', '32', '82', '81', '11', '09', '65']:
        return 'Toulouse'
    else:
        return 'Ailleurs'
df['region'] = df['dep'].apply(size)  
regions = ['Paris', 'Lyon', 'Bordeaux', 'Nantes', 'Toulouse']

# Génératio du fichier pour PowerBI - Colombe
#df.to_csv(r''+path+'indeed-reg.csv')
## pip install openpyxl si besoin
#df.to_excel(r''+path+'indeed-reg.xlsx', sheet_name='indeed', index = False)

# Préparation des données pour les algos
from sklearn.datasets import make_regression
df = pd.concat([df,pd.get_dummies(df['region'])], axis=1)

def base(df, region):
    from sklearn.utils.validation import column_or_1d
    df_region = df[['salary_mean', 'query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi',region]]
    df_region = df_region[df_region[region]==1]
    df_region = df_region.dropna(subset=['salary_mean'], axis=0)

    X = df_region[['query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi']]
    y = df_region[['salary_mean']]
    y = column_or_1d(y, warn=True)
    #print(y.shape)
    return(X, y)


def estime(df, region):
    from sklearn.utils.validation import column_or_1d
    df_region = df[['salary_mean', 'query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi',region]]
    df_region = df_region[df_region[region]==1]

    X = df_region[['query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi']]
    y = df_region[['salary_mean']]
    y = column_or_1d(y, warn=True)
    #print(y.shape)
    return(X, y)

'''
## Vérifictaion du dataset Lyonnais
lyon = pd.DataFrame(base(df,'Lyon')[1], columns=['Lyon'])
lyon.to_excel(r''+path+'lyon.xlsx', sheet_name='indeed', index = False)

paris = pd.DataFrame(base(df,'Paris')[1], columns=['Lyon'])
paris.to_excel(r''+path+'Paris.xlsx', sheet_name='indeed', index = False)

from matplotlib import pyplot

pyplot.plot(lyon, linestyle = 'none', marker = 'o', c = 'lime')
pyplot.xlim(0, lyon.shape[0])
pyplot.ylim(0, 110000)
pyplot.title('Salaire sur Lyon')
pyplot.show()
pyplot.plot(paris, linestyle = 'none', marker = 'o', c = 'lime')
pyplot.xlim(0, paris.shape[0])
pyplot.ylim(0, 180000)
pyplot.title('Salaire sur Paris')
pyplot.show()
'''


def full(df):
    from sklearn.utils.validation import column_or_1d
    df_full = df[['salary_mean', 'query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi', 'Ailleurs',
       'Bordeaux', 'Lyon', 'Nantes', 'Paris', 'Toulouse']]
    df_full = df_full.dropna(subset=['salary_mean'], axis=0)

    X = df_full[['query_developpeur', 'query_business+intelligence',
       'query_data+scientist', 'query_data+analyst', 'contract_temps plein',
       'contract_cdd', 'contract_apprentissage', 'contract_indépendant',
       'contract_stage', 'contract_temps partiel', 'contract_intérim',
       'contract_commission', 'contract_contrat pro', 'contract_cdi', 'Ailleurs', 
       'Bordeaux', 'Lyon', 'Nantes', 'Paris', 'Toulouse']]
    y = df_full[['salary_mean']]
    y = column_or_1d(y, warn=True)
    #print(y.shape)
    return(X, y)

'''
for i in regions :
    X, y = base(df, i)
    print(i, y.shape)
'''


def reglin_salaire(df, region) :
    X, y = base(df, region)
    #X, y = full(df)
    y = y.values.tolist()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train )
    score=model.score(X_test,y_test)
    analyse(X_test, y_test, model)
    return(score)


# Analyse de la regression lineaire obtenue
def analyse(X_test, y_test, model):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    #print('Interception : ',model.intercept_)
    #print('Coefficients : ',model.coef_)
    #print('MAE : ',mean_absolute_error(y_test, y_pred))
    #print('MSE : ',mean_squared_error(y_test, y_pred))
    #print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R² : ', r2)
    print('Adjusted R² : ',1-(1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))


# =============================================================================
# Regression linéaire de base...
# =============================================================================

print()
print('LinearRegression')
for i in regions :
    X, y = base(df, i)
    print(i,'=>')
    print(reglin_salaire(df, i))


'''
# =============================================================================
# GradientBoostingClassifier -- fonctionne pas avec des variables continues !!
# =============================================================================

X, y = base(df, 'Paris')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
'''

# =============================================================================
# GradientBoostingRegressor
# =============================================================================

def gbr_salaire(X, y):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, ensemble
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)
    params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    reg.predict(X_test)
    reg.score(X_test, y_test)
    print(analyse(X_test, y_test, reg))

print()
print('GradientBoostingRegressor')
print('full =>')
X, y = full(df)
gbr_salaire(X, y)
for i in regions :
    X, y = base(df, i)
    print()
    print(i,'=>')
    gbr_salaire(X, y)




'''
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
'''

# =============================================================================
# Regression avec KNN
# =============================================================================

def knn_salaire(X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import neighbors
    from sklearn.model_selection import train_test_split

    X, y = base(df, 'Lyon')
    #X, y = full(df)
    y = y.values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=3)
    # #############################################################################
    # Fit regression model
    n_neighbors = 5

    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X_train, y_train).predict(X_test)

    knn.predict(X_test)
    knn.score(X_test, y_test)
    print(analyse(X_test, y_test, knn))


print()
print('KNeighborsRegressor')
print('full =>')
X, y = full(df)
gbr_salaire(X, y)
for i in regions :
    X, y = base(df, i)
    print()
    print(i,'=>')
    gbr_salaire(X, y)



'''
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()
'''




# =============================================================================
# Modèle OLS - Ordinary Least Squares  --  Meilleur modèle ?
# =============================================================================
## Without a constant
def ols_salaire(X, y):
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=3)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test) # make the predictions by the model
    #print(model.summary())
    predictions = model.predict(X)
    
    return(predictions)


print()
print('full =>')
X, y = full(df)
sal=ols_salaire(X, y)

sal={}
for i in regions :
    X, y = base(df, i)
    print()
    print(i,'=>')
    sal[i]=ols_salaire(X, y)

## Ca me génère un dictionnaire
for cle in sal.keys():
    print(cle)

sal['Paris']


    


# =============================================================================
# Modèle GLSAR
# =============================================================================

def glsar_salaire(X, y):
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=5)

    glsar_model = sm.GLSAR(y_train, X_train, 10)
    glsar_results = glsar_model.iterative_fit(1)
    print(glsar_results.summary())

'''
    model = sm.GLSAR(y_train, X_train, rho=2)
    for i in range(10):
        results = model.fit()
        #print("AR coefficients: {0}".format(model.rho))
        rho, sigma = sm.regression.yule_walker(results.resid, order=model.order)
        model = sm.GLSAR(y_train, X_train, rho)
        print("AR coefficients: {0}".format(model.rho))
'''

print()
print('full =>')
X, y = full(df)
glsar_salaire(X, y)
for i in regions :
    X, y = base(df, i)
    print()
    print(i,'=>')
    glsar_salaire(X, y)



# =============================================================================
# Modèle QuantileRegressor -- Pas trouvé comment ça fonctionne
# =============================================================================
## Nécessite : pip install skquantreg
from skquantreg import QuantileRegressor
import numpy as np

X, y = base(df, 'Paris')
X, y = full(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=5)

model = QuantileRegressor(q=0.5)
model.fit(X_train, y_train)
model.predict(X_test)

model.score(X_test, y_test)


# =============================================================================
# RecursiveLS - Moindres carrés récursifs -- nan en retour...
# =============================================================================
import statsmodels.api as sm
X, y = base(df, 'Lyon')
#X, y = full(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=5)

mod = sm.RecursiveLS(y, X)
res = mod.fit()

print(res.summary())


# =============================================================================
# Weighted Least Squares
# =============================================================================
def wls_salaire(X, y):
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=5)
    mod_wls = sm.WLS(y_train, X_train)
    res_wls = mod_wls.fit()
    print(res_wls.summary())

print()
print('Weighted Least Squares')
print()
print('full =>')
X, y = full(df)
glsar_salaire(X, y)
for i in regions :
    X, y = base(df, i)
    print()
    print(i,'=>',y.shape)
    glsar_salaire(X, y)









