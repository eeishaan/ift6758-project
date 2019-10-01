from sklearn.dummy import DummyRegressor, DummyClassifier

class MajorityClassifier:
    def __init__(self):
        self.clf = DummyClassifier(strategy='most_frequent')

    def fit(self, X, y):
        X = X['user_id']
        self.clf.fit(X, y)

    def predict(self, X):
        X = X['user_id']
        return self.clf.predict(X)

class MeanRegressor:
    def __init__(self):
        self.reg = DummyRegressor(strategy='mean')

    def fit(self, X, y):
        X = X['user_id']
        self.reg.fit(X, y)

    def predict(self, X):
        X = X['user_id']
        return self.reg.predict(X)