import sys

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.append('../')  # TODO fix these imports properly
from models.final_estimator import BaseEstimator


class TreeEnsembleEstimator(BaseEstimator):
    def __init__(self):
        """
        Final classifier used for the gender task. The model is a voting classifier composed of a Random Forest
        classifier, a Extra Trees classifier and a Gradient Boosting classifier.
        """
        super(TreeEnsembleEstimator, self)
        model = VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=50)),
                ('b', ExtraTreesClassifier(n_estimators=50)),
                ('c', GradientBoostingClassifier(n_estimators=100))],
            weights=(1, 1, 2),
            voting='hard',
            n_jobs=4
        )
        self.clf = make_pipeline(StandardScaler(), MinMaxScaler(), model)

    def fit(self, X, y=None):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        X = X['image']

        # filter out non-null rows
        non_null_rows = X.notnull().all(axis=1)
        X, y = X[non_null_rows.values], y[non_null_rows.values]

        self.clf.fit(X, y)
        return self

    def predict(self, X):
        """
         Predicts the categories
         :param X: input dataframe
         :return: list of predicted categories
         """
        X = X['image']
        # if row contains null, we predict 1
        pred = np.ones(shape=(len(X)))

        # filter out non-null rows
        non_null_rows = X.notnull().all(axis=1)
        X_pred = X[non_null_rows]
        pred[non_null_rows] = self.clf.predict(X_pred)

        return pred
