from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from models.final_estimator import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

class PersonalityTreeRegressor(BaseEstimator):
    def __init__(self, n_estimators=200):
        super(PersonalityTreeRegressor, self).__init__()
        self.model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=n_estimators))
    
    def fit(self, X, y):
        X = X['text']
        self.model.fit(X, y)

    def predict(self, X):
        X = X['text']
        return self.model.predict(X)

class PersonalityMLPRegressor(BaseEstimator):
    def __init__(self, hiddens=[10]):
        super(PersonalityMLPRegressor, self).__init__()
        self.model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hiddens))

    def fit(self, X, y):
        X = X['text']
        self.model.fit(X, y)

    def predict(self, X):
        X = X['text']
        return self.model.predict(X)


