import pandas as pd

class SingleTaskEstimator:
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

        pred_df = pd.DataFrame(index=X['user_id'], columns=['age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])

        pred_df['age'] = self.age_clf.predict(X)
        pred_df['gender'] = self.gender_clf.predict(X)

        pred_df['ope'] = self.ope_reg.predict(X)
        pred_df['con'] = self.con_reg.predict(X)
        pred_df['ext'] = self.ext_reg.predict(X)
        pred_df['agr'] = self.agr_reg.predict(X)
        pred_df['neu'] = self.neu_reg.predict(X)