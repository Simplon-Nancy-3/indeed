import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta
import re, json

class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, names=None):
        self.names = names
    
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X[self.names]
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X, y)

class MultiCategoriesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, target, separator=',', replace_key_value = {}, drop_key = [], drop_target=False, clean_func=lambda x: x.strip()):
        self.target = target
        self.separator = separator
        self.clean_func = clean_func
        self.replace_key_value = replace_key_value
        self.drop_key = drop_key
        self.drop_target = drop_target
        self.keys_ = None

    def fit(self, X, y = None):
        return self.fit_series(X[self.target])
    def transform(self, X, y = None):
        return pd.concat([X, pd.DataFrame(
                self.transform_series(X[self.target]), columns=self.format_keys(self.keys_))], 
                axis=1).drop(self.format_keys(self.drop_key) + [self.target] if self.drop_target else [], axis=1)
    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def fit_series(self, X):
        X = X.to_numpy() if isinstance(X, pd.Series) else X
        self.keys_ = set()        

        for x in X:
            for key in str(x).split(','):
                self.keys_.add(self.catch_value(key))
        self.keys_ = list(self.keys_)

        return self
    def transform_series(self, X):
        X = X.to_numpy() if isinstance(X, pd.Series) else X
        res = np.zeros((len(X), len(self.keys_)), dtype=int)

        for i in range(len(X)):
            for key in str(X[i]).split(self.separator):
                res[i, self.keys_.index(self.catch_value(key))] = 1

        return res    
    def fit_transform_series(self, X):
        return self.fit_series(X).transform_series(X)
        # Todo return dataframe if dataframe is passed return numpy array else

    def catch_value(self, key):
        key = self.clean_func(key)
        if key in self.replace_key_value.keys():
            return self.replace_key_value[key]
        return key 
    def format_keys(self, keys):
        return ['{}_{}'.format(self.target, key) for key in keys]

def transform_salary(X, target='salary', 
    names=['salary_origin_mode','salary_min','salary_max','salary_mean'], 
    target_mode='an', hour_by_day=8, day_by_week=5, week_by_month=4,
    month_by_year=12, drop_target=False):

    # Todo : handle conversion elsewhere 
    target_mode = str(target_mode)
    to_year = {
        'None': lambda:x,
        'heure': lambda x: x * hour_by_day *  day_by_week * week_by_month * month_by_year,
        'jour': lambda x: x * day_by_week * week_by_month * month_by_year,
        'semaine': lambda x: x * week_by_month * month_by_year,
        'mois': lambda x: x * month_by_year,
        'an': lambda x: x}
    from_year = {
        'None': lambda x: x,
        'heure': lambda x: x / month_by_year / week_by_month / day_by_week / hour_by_day,
        'jour': lambda x: x / month_by_year / week_by_month / day_by_week,
        'semaine': lambda x: x / month_by_year / week_by_month,
        'mois': lambda x: x / month_by_year,
        'an': lambda x: x}

    x = X[target].to_numpy()
    res = np.empty((len(x), len(names)), dtype=object)

    for i in range(0, len(x)):
        salary = str(x[i])
        salary_min_max = re.findall(r'([0-9 ]+)€', salary)

        if len(salary_min_max) == 0 or not salary_min_max[0][0].isnumeric():
            res[i, 0] = res[i, 1] = res[i, 2] = res[i, 3] = np.nan
        else:
            res[i, 0] = re.search(r'par ([a-z]+)', salary).groups()[0]
            res[i, 1] = res[i, 2] = from_year[target_mode](to_year[res[i, 0]](float(salary_min_max[0].replace(' ', ''))))
            if len(salary_min_max) > 1:
                res[i, 2] = from_year[target_mode](to_year[res[i, 0]](float(salary_min_max[1].replace(' ', ''))))
            res[i, 3] = (res[i, 1] + res[i, 2]) / 2 

    X[names[0]] = res[:, 0]
    X[names[1]] = res[:, 1].astype(float)
    X[names[2]] = res[:, 2].astype(float)
    X[names[3]] = res[:, 3].astype(float)
 
    return X.drop(target, axis=1) if drop_target else X

def transform_date(X, target='day_since', name='date', scrap_day= datetime(2020, 5, 20), drop_target=False):
    x = X[target].to_numpy()
    res = np.empty((len(x)), dtype=object)

    for i in range(0, len(x)):
        _date = re.search(r'([0-9]+)', str(x[i]))
        res[i] =  scrap_day - timedelta(days=int(_date.groups()[0])) if _date else scrap_day

    X[name] = res 

    return X.drop(target, axis=1) if drop_target else X

def transform_rating_mean(X, target='rating_mean', nan_value=0, keep_original=False):
    if keep_original:
        X['%s_original'%target] = X[target]
    X[target] = X[target].apply(lambda x : float(x.replace(',', '.')) if str(x)[0].isdigit() else nan_value)
    return X

def transform_rating_count(X, target='rating_count', nan_value=0, keep_original=False):
    if keep_original:
        X['%s_original'%target] = X[target]
    X[target] = X[target].apply(lambda x : int(x.split(' ')[0].replace(',', '')) if type(x) == str else nan_value)
    return X

def transform_location(X, target='location', names=['dep', 'region'],  nan_value=np.nan, drop_target=False):
    dep_names, dep_regions, dep_nums = get_deps()
    x = X[target].to_numpy()
    res = np.empty((len(x), len(names)), dtype=object)

    for i in range(0, len(x)):
        str_ = str(x[i]).lower().strip()
        dep = re.search(r'\(([^)]+)\)', str_)
        if dep != None:
            res[i, 0] = dep.groups()[0]
            # print('{} - {}'.format(str_, res[i, 0]))
            res[i, 1] = dep_nums[res[i, 0]]
        elif str_ in dep_names.keys():
            res[i, 0] = dep_names[str_]
            # print('{} - {}'.format(str_, res[i, 0]))
            res[i, 1] = dep_nums[res[i, 0]]
        elif str_ in dep_regions:
            # print('{} - {}'.format(str_, res[i, 0]))
            res[i, 1] = str_
    X[names[0]] = res[:, 0]
    X[names[1]] = res[:, 1]

    return X.drop(target, axis=1) if drop_target else X

def get_deps(path='processing/deps.json'):
    names, regions, nums = {}, [], {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for dep in json.loads(f.read()):
            names[dep['dep_name']] = str(dep['num_dep']).zfill(2)
            regions.append(dep['region_name'])
            nums[str(dep['num_dep']).zfill(2)] = dep['region_name']
    return names, set(regions), nums

def transform_sponso(X, target='sponso', keep_original=False):
    if keep_original:
        X['%s_original'%target] = X[target]
    X[target] = X[target].apply(lambda x : 0 if type(x) == str else 1)
    return X

def band_numerical(X, target='salary_mean', name='salary_band', bands = [25000, 60000, 1000000], drop_target=False):
    x = X[target].to_numpy()
    res = np.empty(len(x), dtype=object)
    for i in range(0, len(x)):
        if np.isnan(x[i]):
            continue
        res[i] = len(bands)
        for j in range(0, len(bands)):
            if x[i] <= bands[j]:
                res[i] = j
                break;
    X[name] = res.astype(float)
    return X.drop(target, axis=1) if drop_target else X

indeed_pl = Pipeline(
    steps=[
        ('salary_transformer', FunctionTransformer(transform_salary)),
        ('date_transformer', FunctionTransformer(transform_date)),
        ('query_transfomer', MultiCategoriesTransformer(
            'query',
            replace_key_value= {'devellopeur':'developpeur'},
            clean_func=lambda x: x.replace('"', '').replace('[', '').replace(']', '').lower().strip())),
        ('contract_transformer', MultiCategoriesTransformer(
            'contract', 
            replace_key_value={'freelance / indépendant':'indépendant'},
            drop_key=['nan'],
            clean_func=lambda x: x.lower().strip())),
        ('rating_mean_transformer', FunctionTransformer(transform_rating_mean)),
        ('rating_count_transformer', FunctionTransformer(transform_rating_count)),
        ('location_transformer', FunctionTransformer(transform_location)),
        ('sponso_transformer', FunctionTransformer(transform_sponso)),
        ('salary_bander', FunctionTransformer(band_numerical))
    ],
    verbose=1)
# indeed_pl.fit_transform(pd.read_csv('csv/jobs_it.csv')).info()

    