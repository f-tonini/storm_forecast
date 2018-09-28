from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('regressor', GradientBoostingRegressor(max_depth=7, subsample=0.9, n_estimators=200))
        ])
        
    def fit(self, X, y):
        self.reg.fit(X, y)
    def predict(self, X):
        return self.reg.predict(X)
