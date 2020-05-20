import pandas as pd
import numpy as np

NONE_VALUE = [0,0,0,0]

class SalaryProcessor():
    def __init__(self, by='mois', hour_by_day=8, day_by_week=5, week_by_month=4, month_by_year=12):
        self.by = by
        self.conv_values = [hour_by_day, day_by_week, week_by_month, month_by_year, 1]
        self.conv_names = ['heure', 'jour', 'semaine', 'mois', 'an']
        self.states = [
            lambda x: NONE_VALUE,   # if len == 0
            lambda x: NONE_VALUE,   # if len == 1
            lambda x: [             # if len == 2
                int(x[0].replace(' ', '')), 
                int(x[0].replace(' ', '')), 
                int(x[0].replace(' ', '')),
                self.conv_names.index(x[1].split(' ')[-1])],
            lambda x: [             # if len == 3
                int(x[0].replace(' ', '')), 
                int(x[1].replace(' ', '').replace('-', '')), 
                (int(x[0].replace(' ', '')) + int(x[1].replace(' ', '').replace('-', '')))/2, 
                self.conv_names.index(x[2].strip().split(' ')[-1])]]

    def process_series(self, X):
        X = X.to_numpy() if type(X) == pd.Series else X
        res = []
        for row in X:
            splited = str(row).split('€')
            res.append(np.array(self.states[len(splited)](splited)))
            if res[-1][0] != 0:
                origin_fmt = int(res[-1][3])
                target_fmt = self.conv_names.index(self.by)
                if origin_fmt != target_fmt:
                    step = int((target_fmt - origin_fmt)/ (abs(target_fmt - origin_fmt)))
                    origin_fmt -= 1 if step > 0 else 0
                    while origin_fmt != target_fmt:
                        res[-1][:3] = res[-1][:3] * self.conv_values[origin_fmt + 1] if step > 0 else res[-1][:3] / self.conv_values[origin_fmt - 1]
                        origin_fmt += step
        return res

    def process_dataframe(self, df, target):
        return pd.concat(
            [
                df, 
                pd.DataFrame(self.process_series(df[target]), columns=['salary_min', 'salary_max', 'salary_mean', 'salary_original_mode'])
            ], axis=1).drop(target, axis=1)