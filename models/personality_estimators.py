from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.final_estimator import BaseEstimator


class PersonalityTreeRegressor(BaseEstimator):
    def __init__(self, n_estimators, pca_num=None):
        """
        Final regressor for the persnality traits tasks.
        Currently using a Gradient Boosting model and preprocessing the input by normalizing the input data
        Optionally can also apply dimensionality reduction to the input data
        :param n_estimators: number of estimators for the gradient boosting model
        :param pca_num:  By default, it wont apply PCA.
         If not None, this parameter defines the number of resulting components when applying PCA.
        """
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
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        X = X['text']
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts values on given set of data
        :param X: input dataframe
        :return: list of predictions for the regression task
        """
        X = X['text']
        return self.model.predict(X)
