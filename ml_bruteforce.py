import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pymongo import MongoClient


REGR_METRICS = [
    ('MAE / mean', lambda x, y, z: mean_absolute_error(x, y), lambda x, y: x / y.mean()),
    ('MSE / mean', lambda x, y, z: mean_squared_error(x, y), lambda x, y: x / (y.mean()**2)),
    ('RMSE / mean', lambda x, y, z: mean_squared_error(x, y, squared=False), lambda x, y: x / y.mean()),
    ('R2', lambda x, y, z: r2_score(x, y), lambda x, y: x ),
    ('AR2' , lambda x, y, z: (1-(1-r2_score(x, y))*((len(x)-1)/(len(x)-len(z.columns)-1))), lambda x, y: x )
]

client = MongoClient('mongodb://localhost:27017')
db = client.indeed['ml_region']

class BruteForce:
    def __init__(self, models = [], params_iter=100, row_step=50, metrics=REGR_METRICS, query=[], contract=[], region=[], mode=[]):
        self.models = models
        self.params_iter = params_iter
        self.row_step = row_step
        self.metrics = metrics
        self.query = query
        self.contract = contract
        self.region = region
        self.mode = mode


    def fit_region(self, df):
        self.current_region = 0
        regions = ['île-de-france','auvergne-rhône-alpes','pays de la loire',
            'occitanie','nouvelle-aquitaine','provence-alpes-côte d\'azur',
            'hauts-de-france','bretagne','grand est']
        self.it_max = len(self.models) * len(regions)
        features = self.contract + self.mode
        self.res = {}
        print('try {} models for features {}'.format(self.it_max, features))

        for region in regions:
            self.res['region'] = region
            sample = df[df['region_%s'%region] == 1]
            X_train, X_test, y_train, y_test = train_test_split(sample[features], sample['salary_mean'], test_size=.3, random_state=0)
            self.fit_models(X_train, X_test, y_train, y_test)
            self.current_region += 1
    def fit_models(self, X_train, X_test, y_train, y_test):
        self.current_model = 0
        for name, model, params in self.models:
            self.res['model'] = name
            model_exist = db.find_one({'region': self.res['region'], 'model' : self.res['model']})
            if not model_exist:
                model = model if params == None else self.params_search(model, params, X_train, y_train)
                metric = self.row_search(model, X_train, X_test, y_train, y_test)
                self.res['_id'] = db.count_documents({})
                print(self.res)
                db.insert_one(self.res)
                self.current_model += 1
                it = self.current_region * len(self.models) + self.current_model
                print('{}/{} - {:2f} - {} - {}'.format(it, self.it_max, it / self.it_max * 100, self.res['region'], self.res['model']))

    def params_search(self, model, params, X, y):
        if params == None:
            self.res['params'] = {}
            return model
        search = RandomizedSearchCV(model, params, cv=3, n_iter = self.params_iter, verbose=1, n_jobs=4).fit(X, y)
        self.res['params'] = search.best_params_
        if self.res['model'] == 'KNeighborsRegressor':
            self.res['params']['n_neighbors'] = int(self.res['params']['n_neighbors'] )
        return search.best_estimator_
    def row_search(self, model, X_train, X_test, y_train, y_test):
        search = np.empty((int(np.ceil(len(y_train) / self.row_step)), len(self.metrics)))
        i = 0
        while i < len(y_train):
            i = min(i + self.row_step, len(X_train))
            y_pred = model.fit(X_train[:i], y_train[:i]).predict(X_test)
            for j in range(len(self.metrics)):
                search[int(i/self.row_step), j] = self.metrics[j][1](y_test, y_pred, X_test)
        self.res['row'] = search.tolist()

df = pd.read_csv('csv/jobs_it_process.csv')
df = df.dropna(subset=['salary_mean', 'region'])[ ['salary_origin_mode'] + list(df.columns[16:35])]
df = pd.concat([df, pd.get_dummies(df.pop('region'), prefix='region')], axis=1)
df = pd.concat([
    df, pd.get_dummies(df.pop('salary_origin_mode'), prefix='mode')], axis=1).drop(['dep', 'date'], axis=1)
df.info()

# X.info()

BruteForce([
    ('LinearRegression', LinearRegression(), None),
    ('SGDRegressor', SGDRegressor(random_state=0), {
        'loss':['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
        'penalty':['l2', 'l1', 'elasticnet'],
        'alpha':[.01, .001, .0001, .00001],
        'l1_ratio':[0.1, .15, .5, .75, .9]}),
    ('Ridge', Ridge(random_state=0), {
        'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}),
    ('RandomForestRegressor', RandomForestRegressor(random_state=0), {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=0), {
        'loss': ['ls', 'lad', 'huber', 'quantile'],    
        'criterion': ['friedman_mse', 'mse', 'mae'],
        'warm_start':[True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'subsample': [.6, .8, 1],
        'n_estimators':range(20,81,10}),
    ('XGBRegressor', XGBRegressor(random_state=0), {
        'objective': ['reg:squarederror','reg:gamma','reg:tweedie'],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]}),
    ('AdaBoostRegressor', AdaBoostRegressor(random_state=0), {
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'loss': ['linear', 'square', 'exponential']}),
    ('KNeighborsRegressor', KNeighborsRegressor(), {
        'n_neighbors': np.arange(1, 25),
        'weights':['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
    ('LinearSVR', LinearSVR(random_state=0), {
        'C':[.25, .5, .75, 1, 1.25, 1.5, 1.75, 2]}),
    ('NuSVR', NuSVR(), {
        'nu':[.25, .5, .75],
        'C':[.25, .5, .75, 1, 1.25, 1.5, 1.75, 2],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}),
    ('SVR', SVR(), {
        'C':[.25, .5, .75, 1, 1.25, 1.5, 1.75, 2],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    })
    ], 
    query=list(df.columns[1:5]), contract=list(df.columns[5:16]), 
    region=list(df.columns[16:29]), mode=list(df.columns[29:])).fit_region(df)

client.close()

