from sklearn.ensemble import RandomForestRegressor
from models.final_estimator import BaseEstimator
from sklearn.externals import joblib

class PersonalityTreeRegressor(BaseEstimator):
    def __init__(self, n_estimators=200):
        super(PersonalityTreeRegressor, self).__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.predict(X)


