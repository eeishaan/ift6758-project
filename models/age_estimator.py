import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight

from models.final_estimator import BaseEstimator
from utils.label_mapping import age_to_age_group


class AgeEstimator(BaseEstimator):
    def __init__(self, n_estimators=1000, pca_components=10):
        super().__init__()
        # TODO : This is only a example, we can modify the model here
        self.clf = VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=n_estimators)),
                ('b', ExtraTreesClassifier(n_estimators=n_estimators)),
                ('c', GradientBoostingClassifier(n_estimators=n_estimators))],
            weights=(1, 1, 1),
            voting='hard',
            n_jobs=4
        )
        self.std_scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)

    def fit_pca(self, X):
        X = np.array(X)
        self.pca.fit(X)
        print(self.pca.explained_variance_.round(2))

    def fit(self, X, y=None):
        age_to_group_map = np.vectorize(age_to_age_group)
        y = age_to_group_map(y)
        X = self._preprocess(X)
        non_null_rows = X.notnull().all(axis=1)
        X, y = X[non_null_rows.values], y[non_null_rows.values]
        self.fit_pca(X)
        X = self.pca.transform(X)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = self._preprocess(X)
        pred = np.full(shape=(len(X)), fill_value="xx-24")  # ["xx-24", "25-34", "35-49", "50-xx"]
        non_null_rows = X.notnull().all(axis=1)
        X_pred = X[non_null_rows]
        X_pred = self.pca.transform(X_pred)
        pred[non_null_rows] = self.clf.predict(X_pred)

        return pred

    def _normalize_data(self, data):
        scaled_features = self.std_scaler.fit_transform(data.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
        return scaled_features_df

    def _preprocess(self, X):
        # TODO: for now it only includes the image and the relation, but it is easy to add the text data
        image_data = X["image"]
        input_data = X["relation"].set_index('userId')
        input_data = image_data.join(input_data, on="userId")
        input_data = self._normalize_data(input_data)
        input_data = input_data.join(X["text"].set_index("userId"), on="userId")

        return input_data

    def get_like_count_per_user(self, relational_data):
        relational_data = relational_data.drop(columns=relational_data.columns[0])
        new_columns_name = ["userId", relational_data.columns[1]]
        relational_data.columns = new_columns_name
        relational_data["like_count"] = relational_data.groupby('userId')['userId'].transform('count')
        relational_data = relational_data.drop(columns="like_id")
        return relational_data

    def save(self, output_path):
        joblib.dump(self, output_path)
