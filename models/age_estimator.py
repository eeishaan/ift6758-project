from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from models.final_estimator import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from models.baselines import MajorityClassifier
from sklearn.externals import joblib


class AgeClassifier(BaseEstimator):
    def __init__(self, n_estimators=200):
        super(AgeClassifier, self).__init__()
        self.clf = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=n_estimators))
        self.fallback_clf = MajorityClassifier()

    def fit(self, X, y):
        X = X['image']

        # filter out non-null rows
        non_null_rows = X.notnull().all(axis=1)
        X, y = X[non_null_rows.values], y[non_null_rows.values]

        self.clf.fit(X, y)
        self.fallback_clf.fit(X, y)
        return self

    def predict(self, X):
        X = X['image']
        # if row contains null, we predict fallback model's classification
        pred = self.fallback_clf.predict(X)

        # filter out non-null rows
        non_null_rows = X.notnull().all(axis=1)
        X_pred = X[non_null_rows]
        pred[non_null_rows] = self.clf.predict(X_pred)

        return pred


