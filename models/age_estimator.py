import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import compute_sample_weight

from models.gender_estimator import TreeEnsembleEstimator


class AgeEstimator(TreeEnsembleEstimator):
    def __init__(self):
        super().__init__()
        self.clf = VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=50)),
                ('b', ExtraTreesClassifier(n_estimators=50)),
                ('c', GradientBoostingClassifier(n_estimators=100))],
            weights=(1, 2, 1),
            voting='hard',
            n_jobs=4
        )

    def fit(self, X, y=None):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        X_train = self._process(X, is_train=True)
        self.clf.fit(X_train, y, sample_weight=sample_weight)
        return self

    def save(self, output_path):
        joblib.dump(self, output_path)
