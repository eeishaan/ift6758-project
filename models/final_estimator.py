import abc

import pandas as pd
from sklearn.externals import joblib
from utils.scoring import *


class BaseEstimator(abc.ABC):
    """
    Base class for all estimators
    """
    @abc.abstractmethod
    def fit(self, X, y):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """
        Predicts the values with given input data
        :param X: input dataframe
        :return: list of predicted values
        """
        pass

    def save(self, output_path):
        """
        Saves the model
        :param output_path: model output location
        :return:
        """
        joblib.dump(self, output_path)


class SingleTaskEstimator(BaseEstimator):
    def __init__(
        self, gender_clf, age_clf, ope_reg, con_reg, ext_reg, agr_reg, neu_reg
    ):
        """
        Model composed of specific sub model for each task
        :param gender_clf: Gender classifier model of type BaseEstimator
        :param age_clf: Age classifier model  of type BaseEstimator
        :param ope_reg: Openness regression model  of type BaseEstimator
        :param con_reg: Conscientiousness regression model of type BaseEstimator
        :param ext_reg: Extraversion regression model of type BaseEstimator
        :param agr_reg: Agreeableness regression model of type BaseEstimator
        :param neu_reg: Neuroticism regression model of type BaseEstimator
        """
        # Independent task models
        self.neu_reg = neu_reg
        self.agr_reg = agr_reg
        self.ext_reg = ext_reg
        self.con_reg = con_reg
        self.age_clf = age_clf
        self.ope_reg = ope_reg
        self.gender_clf = gender_clf

    def fit(self, X, y):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        self.gender_clf.fit(X, y["gender"])
        print("Trained gender classifier.")
        self.ope_reg.fit(X, y["ope"])
        print("Trained ope classifier.")
        self.con_reg.fit(X, y["con"])
        print("Trained con classifier.")
        self.ext_reg.fit(X, y["ext"])
        print("Trained ext classifier.")
        self.agr_reg.fit(X, y["agr"])
        print("Trained agr classifier.")
        self.neu_reg.fit(X, y["neu"])
        print("Trained neu classifier.")
        self.age_clf.fit(X, y["age"])
        print("Trained age classifier.")

    def predict(self, X):
        """
        Predicts the values with given input data
        :param X: input dataframe
        :return: list of predicted values
        """
        pred_df = pd.DataFrame(
            index=X["user_id"],
            columns=["age", "gender", "ope", "con", "ext", "agr", "neu"],
        )

        pred_df["gender"] = self.gender_clf.predict(X)
        pred_df["ope"] = self.ope_reg.predict(X)
        pred_df["con"] = self.con_reg.predict(X)
        pred_df["ext"] = self.ext_reg.predict(X)
        pred_df["agr"] = self.agr_reg.predict(X)
        pred_df["neu"] = self.neu_reg.predict(X)
        pred_df["age"] = self.age_clf.predict(X)

        return pred_df

    def eval(self, Xtest, ytest, save=False, age_to_group=True):
        """
        Evaluates the model on a given test set
        :param Xtest: test input data
        :param ytest: labels data for all task
        :param save: if True, saves the evalutaion results
        :param age_to_group: if True, converts age values to category names
        :return: evaluation results for all tasks as a dictionary
        """
        ypred = self.predict(Xtest)
        eval_results = {
            "gender": gender_score(ypred["gender"], ytest["gender"]),
            "ope": personality_score(ypred["ope"], ytest["ope"]),
            "con": personality_score(ypred["con"], ytest["con"]),
            "ext": personality_score(ypred["ext"], ytest["ext"]),
            "agr": personality_score(ypred["agr"], ytest["agr"]),
            "neu": personality_score(ypred["neu"], ytest["neu"]),
            "age": age_score(ypred["age"], ytest["age"], age_to_group),
        }
        if save:
            pd.to_pickle(eval_results, "evaluation_results.pkl")

        print(
            "The age accuracy is: {}\n \
              The gender accuracy is: {}\n \
              The ope rmse is: {}\n)\
              The con rmse is: {}\n)\
              The ext rmse is: {}\n)\
              The agr rmse is: {}\n)\
              The neu rmse is: {}\n".format(
                eval_results["age"],
                eval_results["gender"],
                eval_results["ope"],
                eval_results["con"],
                eval_results["ext"],
                eval_results["agr"],
                eval_results["neu"],
            )
        )
        return eval_results
