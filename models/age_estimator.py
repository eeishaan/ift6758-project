import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from utils.label_mappings import category_id_to_age
from xgboost import XGBClassifier
from models.final_estimator import BaseEstimator


class AgeEstimator(BaseEstimator):
    def __init__(
            self,
            n_estimators=1000,
            text_svd_components=None,
            image_svd_components=None,
            num_ensemble=3,
            normalize_likeid_age_distrbn=True,
            minimum_like_counts=5,
            related_models=[],
    ):
        """
        Final classifier used for the age task. The model is an ensemble classifier composed of gradient boosting models
        :param n_estimators: Number of gradient boosting estimators
        :param text_svd_components: if not None, applies svd dimensionality reduction with given number of components
        for the text data. Defaults with no dimensionality reduction.
        :param image_svd_components:i f not None, applies svd dimensionality reduction with given number of components
        for the image data. Defaults with no dimensionality reduction.
        :param num_ensemble: Number of models in the ensemble classifier
        :param normalize_likeid_age_distrbn: if True, we normalize the age-like distribution
        :param minimum_like_counts: Minimum like required for a page to be considered in the mean inverse
        age-like count distribution computation
        :param related_models: if we provide a list of related models (ex: gender model), add the prediction of these
        models as additional input of this model. Defaults with empty list.
        """
        super().__init__()
        self.related_models = related_models
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
        self.text_svd_model = None
        self.image_svd_model = None
        if text_svd_components is not None:
            self.text_svd_model = TruncatedSVD(n_components=text_svd_components)
        if image_svd_components is not None:
            self.image_svd_model = TruncatedSVD(n_components=image_svd_components)
        self.std_scaler = StandardScaler()
        self.age_idx_to_age_group_func = np.vectorize(category_id_to_age)
        self.normalize_likeid_age_distrbn = normalize_likeid_age_distrbn
        self.minimum_like_counts = minimum_like_counts

    def fit(self, X, y=None):
        """
        Fits the model
        :param X: input dataframe
        :param y: labels dataframe
        :return:
        """
        like_age = pd.merge(
            X["relation"], y.to_frame(), left_index=True, right_index=True
        )[["like_id", "age"]]
        like_ages_counts = (
            like_age.groupby(["like_id", "age"]).size().unstack(fill_value=0)
        )
        self.like_ages_counts = like_ages_counts[like_ages_counts.sum(axis=1) >= self.minimum_like_counts]
        if self.normalize_likeid_age_distrbn:
            self.like_ages_counts = self.like_ages_counts.div(
                self.like_ages_counts.sum(axis=1), axis=0
            )
        X = self._preprocess(X, is_train_mode=True)
        sample_weight = None
        self.clf.fit(X.values, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Predicts the age categories with given input data
        :param X: input dataframe
        :return: list of predicted values
        """
        X = self._preprocess(X, is_train_mode=False)
        pred = self.clf.predict(X.values)
        pred = self.age_idx_to_age_group_func(pred)
        return pred

    def _normalize_data(self, data):
        """
        Normalize the data
        :param data: data to normalize
        :return: normalized data
        """
        scaled_features = self.std_scaler.fit_transform(data.values)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=data.index, columns=data.columns
        )
        return scaled_features_df

    def _preprocess(self, X, is_train_mode=False):
        """
        Preprocess  by computing the mean inverse age-like count distribution and by adding the age count per user along
        other normalized data sources into a single dataframe,
        :param X: Input data
        :param is_train_mode: If True and if we the svd models are not None, will train SVD models when fitting the
        model
        :return: Preprocessed data
        """

        image_data = X["image"]
        image_data = image_data.fillna(image_data.mean())
        image_data = self._normalize_data(image_data)

        like_count_data = self.get_num_likes(X["relation"])
        like_count_data = self.std_scaler.fit_transform(like_count_data)
        # mean of inverse age distributions for page likes
        mean_inverse_age_distribution_data = (
            pd.merge(
                X["relation"],
                self.like_ages_counts,
                left_on="like_id",
                right_index=True,
            ).iloc[:, -4:].groupby("userId").mean()
        )
        # rearrange the index as per the input
        mean_inverse_age_distribution_data = mean_inverse_age_distribution_data.reindex(
            X["image"].index
        )

        text_data = X["text"]
        text_data = self._normalize_data(text_data)

        text_data = self.apply_svd_dimensionality_reduction(text_data, self.text_svd_model, is_train_mode)
        image_data = self.apply_svd_dimensionality_reduction(image_data, self.image_svd_model, is_train_mode)

        input_data = pd.concat(
            [
                pd.DataFrame(image_data.values),
                pd.DataFrame(like_count_data),
                pd.DataFrame(mean_inverse_age_distribution_data.values),
                pd.DataFrame(text_data.values),
            ],
            axis=1,
        )
        input_data = self.add_other_task_prediction(X, input_data)
        return input_data

    @staticmethod
    def apply_svd_dimensionality_reduction(data, svd_model, is_train_mode):
        """
        Conditionally applies svd dimensionality reduction on input data
        :param data: Data which will be transformed
        :param svd_model: SVD model of type TruncatedSVD from scikit
        :param is_train_mode: If True, will also train the svd model
        :return: Transformed data
        """
        if svd_model is not None:
            if is_train_mode:
                svd_model.fit(data)
                print(f"text_pca explained ratio {svd_model.explained_variance_ratio_.cumsum()}")
            data = svd_model.transform(data)
        return data

    def add_other_task_prediction(self, X, input_data):
        """
        Adds other models predictions as additional features into given data
        :param X: Original input data (not preprocessed)
        :param input_data: Current input data
        :return: Current input data with other models predictions
        """

        if len(self.related_models) > 0:
            for i, clf in enumerate(self.related_models):
                if clf is not None:
                    pred = clf.predict(X)
                    input_data[f"pred_{i}"] = pred
        return input_data

    @staticmethod
    def get_num_likes(relation_data):
        """
        From the relation dataframe, get like count per user
        :param relation_data: Relational dataframe
        :return: List of like count for each user
        """
        relation_data_group = relation_data.groupby(["userId"])
        like_count_per_user = np.zeros(shape=(len(relation_data_group), 1))
        for i, (user_id, relation_data) in enumerate(relation_data_group):
            like_count_per_user[i] = len(relation_data["like_id"])
        return like_count_per_user
