from sklearn.dummy import DummyRegressor, DummyClassifier


class MajorityClassifier:
    def __init__(self):
        self.clf = DummyClassifier(strategy='most_frequent')

    def fit(self, X, y):
        try:
            X = X['user_id']
        except (IndexError, KeyError) as e:
            pass
        self.clf.fit(X, y)

    def predict(self, X):
        try:
            X = X['user_id']
        except (IndexError, KeyError) as e:
            pass
        return self.clf.predict(X)


class MeanRegressor:
    def __init__(self):
        self.reg = DummyRegressor(strategy='mean')

    def fit(self, X, y):
        try:
            X = X['user_id']
        except (IndexError, KeyError) as e:
            pass
        self.reg.fit(X, y)

    def predict(self, X):
        try:
            X = X['user_id']
        except (IndexError, KeyError) as e:
            pass
        return self.reg.predict(X)
