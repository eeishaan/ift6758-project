import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)

sys.path.append('../')  # TODO fix these imports properly
from models.final_estimator import BaseEstimator


class TreeEnsembleEstimator(BaseEstimator):
    def __init__(self):
        self.clf = VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=150)),
                ('b', ExtraTreesClassifier(n_estimators=150)),
                ('c', GradientBoostingClassifier(n_estimators=150))],
            weights=(1, 1, 2),
            voting='hard',
            n_jobs=4
        )

    def fit(self, X, y=None):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
