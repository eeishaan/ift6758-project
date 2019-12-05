import time
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier

from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler

from utils.label_mappings import category_id_to_age
from xgboost import XGBClassifier

from models.final_estimator import BaseEstimator


class AgeEstimator(BaseEstimator):
    def __init__(
        self,
        n_estimators=1000,
        text_svd_components=50,
        image_svd_components=3,
        num_ensemble=3,
        gender_clf=None,
        ope_reg=None,
        con_reg=None,
        ext_reg=None,
        agr_reg=None,
        neu_reg=None,
    ):
        super().__init__()
        # self.related_model = [gender_clf, ope_reg, con_reg, ext_reg, agr_reg, neu_reg]
        self.related_model = []
        estimators = []
        for i in range(num_ensemble):
            estimators.append(
                (
                    str(i),
                    XGBClassifier(
                        n_estimators=n_estimators,
                        n_jobs=10,
                        learning_rate=0.01,
                        max_depth=3,
                        colsample_bytree=1,
                    ),
                )
            )

        self.clf = VotingClassifier(estimators=estimators, voting="hard", n_jobs=1)
        self.text_pca = TruncatedSVD(n_components=text_svd_components)
        self.image_pca = TruncatedSVD(n_components=image_svd_components)
        self.std_scaler = StandardScaler()
        self.age_idx_to_age_group_func = np.vectorize(category_id_to_age)

    def fit(self, X, y=None):
        X = self._preprocess(X, is_train_mode=True)
        sample_weight = None
        self.clf.fit(X.values, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = self._preprocess(X, is_train_mode=False)
        pred = self.clf.predict(X.values)
        pred = self.age_idx_to_age_group_func(pred)
        return pred

    def _normalize_data(self, data):
        scaled_features = self.std_scaler.fit_transform(data.values)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=data.index, columns=data.columns
        )
        return scaled_features_df

    def _preprocess(self, X, is_train_mode=False):
        image_data = X["image"]
        image_data = image_data.fillna(image_data.mean())
        image_data = self._normalize_data(image_data)

        relation_data = self.get_num_likes(X["relation"])
        relation_data = self.std_scaler.fit_transform(relation_data)

        text_data = X["text"]
        text_data = self._normalize_data(text_data)
        # if is_train_mode:
        #     self.text_pca.fit(text_data)
        #     print(f"text_pca explained ratio {self.text_pca.explained_variance_ratio_.cumsum()}")
        #     self.image_pca.fit(image_data)
        #     # print(f"image_pca explained ratio {self.image_pca.explained_variance_ratio_.cumsum()}")
        #
        # text_data = self.text_pca.transform(text_data)
        # image_data = self.image_pca.transform(image_data)

        input_data = pd.concat(
            [
                pd.DataFrame(image_data.values),
                pd.DataFrame(relation_data),
                pd.DataFrame(text_data.values),
            ],
            axis=1,
        )
        input_data = self.add_other_task_prediction(X, input_data)
        return input_data

    def add_other_task_prediction(self, X, input_data):
        if len(self.related_model) > 0:
            for i, clf in enumerate(self.related_model):
                if clf is not None:
                    pred = clf.predict(X)
                    input_data[f"pred_{i}"] = pred
        return input_data

    def save(self, output_path):
        joblib.dump(self, output_path)

    def get_num_likes(self, relation_data):
        relation_data_group = relation_data.groupby(["userId"])
        like_count_per_user = np.zeros(shape=(len(relation_data_group), 1))
        for i, (user_id, relation_data) in enumerate(relation_data_group):
            like_count_per_user[i] = len(relation_data["like_id"])
        return like_count_per_user
