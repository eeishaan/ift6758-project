from sklearn.dummy import DummyRegressor, DummyClassifier

from models.final_estimator import BaseEstimator


class MajorityClassifier(BaseEstimator):
    def __init__(self):
        """
        Model predicting the most frequent class
        """
        self.clf = DummyClassifier(strategy='most_frequent')

    def fit(self, X, y):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        X = X['user_id']
        self.clf.fit(X, y)

    def predict(self, X):
        """
        Predicts the values with given input data
        :param X: input dataframe
        :return: list of predicted values
        """
        X = X['user_id']
        return self.clf.predict(X)


class MeanRegressor(BaseEstimator):
    def __init__(self):
        """
        Model predicting the mean value of the target label
        """
        self.reg = DummyRegressor(strategy='mean')

    def fit(self, X, y):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        X = X['user_id']
        self.reg.fit(X, y)

    def predict(self, X):
        """
        Predicts the values with given input data
        :param X: input dataframe
        :return: list of predicted values
        """
        X = X['user_id']
        return self.reg.predict(X)
