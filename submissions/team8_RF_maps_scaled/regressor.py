from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('regressor', RandomForestRegressor(n_estimators = 100, max_features=0.2, min_samples_leaf = 20))
        ])

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
