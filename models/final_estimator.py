import abc

import pandas as pd
from sklearn.externals import joblib
from utils.scoring import *


class BaseEstimator(abc.ABC):

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    def save(self, output_path):
        joblib.dump(self, output_path)



class SingleTaskEstimator(BaseEstimator):
    def __init__(self, gender_clf, age_clf, ope_reg, con_reg, ext_reg, agr_reg, neu_reg):
        # Independent task models
        self.neu_reg = neu_reg
        self.agr_reg = agr_reg
        self.ext_reg = ext_reg
        self.con_reg = con_reg
        self.age_clf = age_clf
        self.ope_reg = ope_reg
        self.gender_clf = gender_clf

    def fit(self, X, y):
        self.age_clf.fit(X, y['age'])
        self.gender_clf.fit(X, y['gender'])

        self.ope_reg.fit(X, y['ope'])
        self.con_reg.fit(X, y['con'])
        self.ext_reg.fit(X, y['ext'])
        self.agr_reg.fit(X, y['agr'])
        self.neu_reg.fit(X, y['neu'])

    def predict(self, X):
        pred_df = pd.DataFrame(index=X['user_id'], columns=[
                               'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])

        pred_df['age'] = self.age_clf.predict(X)
        pred_df['gender'] = self.gender_clf.predict(X)

        pred_df['ope'] = self.ope_reg.predict(X)
        pred_df['con'] = self.con_reg.predict(X)
        pred_df['ext'] = self.ext_reg.predict(X)
        pred_df['agr'] = self.agr_reg.predict(X)
        pred_df['neu'] = self.neu_reg.predict(X)

        return pred_df

    def eval(self, Xtest, ytest, save=False):
        ypred = self.predict(Xtest)
        eval_results = {
            'age' : age_score(ypred['age'], ytest['age']),
            'gender' : gender_score(ypred['gender'], ytest['gender']),
            'ope': personality_score(ypred['ope'], ytest['ope']),
            'con': personality_score(ypred['con'], ytest['con']),
            'ext': personality_score(ypred['ext'], ytest['ext']),
            'agr': personality_score(ypred['agr'], ytest['agr']),
            'neu': personality_score(ypred['neu'], ytest['neu'])
        }
        if save:
            pd.to_pickle(eval_results, 'evaluation_results.pkl')

        print("The age accuracy is: {}\n \
              The gender accuracy is: {}\n \
              The ope mse is: {}\n)\
              The con mse is: {}\n)\
              The ext mse is: {}\n)\
              The agr mse is: {}\n)\
              The neu mse is: {}\n".format(eval_results['age'],
                                           eval_results['gender'],
                                           eval_results['ope'],
                                           eval_results['con'],
                                           eval_results['ext'],
                                           eval_results['agr'],
                                           eval_results['neu'],))

    def save(self, output_path):
        joblib.dump(self, output_path)
