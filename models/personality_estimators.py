from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.final_estimator import BaseEstimator


class PersonalityTreeRegressor(BaseEstimator):
    def __init__(self, n_estimators, pca_num):
        super(PersonalityTreeRegressor, self).__init__()
        if pca_num:
            self.model = make_pipeline(
                StandardScaler(),
                PCA(pca_num),
                GradientBoostingRegressor(n_estimators=n_estimators))
        else:
            self.model = make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(n_estimators=n_estimators))

    def fit(self, X, y):
        X = X['text']
        self.model.fit(X, y)

    def predict(self, X):
        X = X['text']
        return self.model.predict(X)


class MultiTaskRegressor(BaseEstimator):
    def __init__(self):
        super(MultiTaskRegressor, self).__init__()
        self.model = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            PCA(n_components=30),
            MultiTaskElasticNet()
        )

    def fit(self, X, y):
        X = X['text']
        self.model.fit(X, y)

    def predict(self, X):
        X = X['text']
        return self.model.predict(X)
