import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append('../')  # TODO fix these imports properly
from models.final_estimator import BaseEstimator
from util.data_processing import preprocess



class TreeEnsembleEstimator(BaseEstimator):
    def __init__(self):
        super(TreeEnsembleEstimator, self)
        self.clf = VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=50)),
                ('b', ExtraTreesClassifier(n_estimators=50)),
                ('c', GradientBoostingClassifier(n_estimators=100))],
            weights=(1, 1, 2),
            voting='hard',
            n_jobs=4
        )

    def _process(self, X, is_train=False):
        standard_scalers = getattr(self, 'standard_scalers', None)
        norm_scalers = getattr(self, 'norm_scalers', None)
        if is_train:
            standard_scalers = None
            norm_scalers = None

        # standardize train
        X_std_train, self.standard_scalers = preprocess(
            X, StandardScaler, standard_scalers)

        # normalize train
        X_norm_train, self.norm_scalers = preprocess(
            X_std_train, MinMaxScaler, norm_scalers)
        return X_norm_train

    def fit(self, X, y=None):
        X_train = self._process(X, is_train=True)
        self.clf.fit(X_train, y)
        return self

    def predict(self, X):
        # if row contains null, we predict 1
        pred = np.ones(shape=(len(X)))

        # filter out non-null rows
        non_null_rows = X.notnull().all(axis=1)
        filtered_X = X[non_null_rows]
        X_pred = self._process(filtered_X)
        pred[non_null_rows] = self.clf.predict(X_pred)

        return pred

    def save(self, output_path):
        joblib.dump(self, output_path)
