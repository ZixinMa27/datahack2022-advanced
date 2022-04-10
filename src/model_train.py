"""
store models 
"""

# import packages 
import os 
import numpy as np
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

# ============= concat features ===========

class Model:
    def __init__(self, X_train, y_train, params: dict={}) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.params = params
        self.name = self.__class__.__name__

    def fit(self): 
        """train model"""
        raise NotImplementedError()
    
    def predict(self, X_test):
        """ predict train/test """
        self.X_test = X_test 
        
    def evaluate(self, X_test, y_test):
        """ report metrics """
        # TODO: report accuracy, F1, and possibly weighted F1
        predictions = self.predict(X_test)
        weights = self.assign_weight(y_test)
        y_test_np = np.array(y_test)

        acc = round(accuracy_score(y_test_np, predictions),4)
        acc_weighted = round(((y_test_np== predictions) * weights).mean(),4)
        f1_score_macro = round(f1_score(y_test_np, predictions, average = 'macro'),4)
        f1_score_weighted = round(f1_score(y_test_np, predictions, average = 'weighted'),4)
        measurement = ['accuracy','accuracy_weighted','f1_score_macro','f1_score_weighted']
        percent = [acc, acc_weighted, f1_score_macro, f1_score_weighted]
        return (dict(zip(measurement, percent)))
    
    # ====== auxiliary =======
    def assign_weight(self, y):
        # TODO: explore different weighting mechanisms 
        array = []
        for i in y:
            if i == 1:
                array.append(1.5)
            elif i == 0:
                array.append(0.7)
            else:
                array.append(1)
        array = np.array(array)
        return array

# ============= linear based ==============
class LogisticRegressionCustom(Model):

    name = 'LogisticRegression'

    def __init__(self, X_train, y_train, params: dict={}) -> None:
        super().__init__(X_train, y_train, params)
        self.model = LogisticRegression(**params)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X_test):
        super().predict(X_test)
        return self.model.predict(X_test)
    

# ============= bagging ==============
class RandomForestClassifierCustom(Model):

    name = 'RandomForestClassification'

    def __init__(self, X_train, y_train, params: dict = {}) -> None:
        super().__init__(X_train, y_train, params)
        self.model = RandomForestClassifier(**params, n_jobs=-1)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        super().predict(X_test)
        return self.model.predict(X_test)

# ============= boosting =============

class XGBoostClassifierCustom(Model):

    name = 'XGBoostClassification'

    def __init__(self, X_train, y_train, params: dict = {}) -> None:
        super().__init__(X_train, y_train, params)
        self.model = xgb.XGBClassifier(**params, n_jobs=-1)
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        super().predict(X_test)
        return self.model.predict(X_test)

class LightGBMClassifierCustom(Model):

    name = 'LightGBMClassification'

    def __init__(self, X_train, y_train, params: dict = {}) -> None:
        super().__init__(X_train, y_train, params)
        self.model = lgb.LGBMClassifier(**params, n_jobs=-1)
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        super().predict(X_test)
        return self.model.predict(X_test)
