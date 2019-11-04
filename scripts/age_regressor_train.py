import pickle

import math
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from models.voting_regressor import VotingRegressor
from util.data_processing import normalize_splits, remove_empty_data
from util.k_fold import k_fold
from util.label_mapping import age_to_age_group, age_group_to_category_id


def eval_regressor_model(model, train_data, test_data, is_log_age=True):
    # fit model
    model.fit(*train_data)
    train_labels = train_data[1]
    test_labels = test_data[1]
    train_pred = model.predict(train_data[0])
    test_pred = model.predict(test_data[0])
    if is_log_age:
        train_labels = np.exp(train_labels)
        test_labels = np.exp(test_labels)
        train_pred = np.exp(train_pred)
        test_pred = np.exp(test_pred)

    age_to_group_map = np.vectorize(age_to_age_group)
    # evaluate train perf
    train_acc = accuracy_score(age_to_group_map(train_labels), age_to_group_map(train_pred))

    # test performance
    test_acc = accuracy_score(age_to_group_map(test_labels), age_to_group_map(test_pred))

    return {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }


class ModeModel(object):
    def __init__(self):
        self.ans = 1

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return len(X) * [self.ans]

    def fit_predict(self, X, y=None):
        return self.transform(X)


def k_fold_train():
    # read data
    DATA_PATH = "../../new_data/Train/"
    pca_model = pickle.load(open("out.pca", 'rb'))
    image_data = pd.read_csv(os.path.join(DATA_PATH, "Image", "oxford.csv"))
    liwc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "liwc.csv"))
    nrc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "nrc.csv"))
    nrc_data.rename(columns={'anger': 'anger_1'},
                    inplace=True)
    input_data = image_data.join(liwc_data.set_index('userId'), on="userId")
    # input_data = input_data.join(nrc_data.set_index('userId'), on="userId")

    profile_data = pd.read_csv(os.path.join(DATA_PATH, "Profile", "Profile.csv"))

    non_empty = remove_empty_data(input_data, profile_data)
    X = []
    y = []
    id_col = ['userId', 'faceID']
    # id_col = ['userId']
    for row in profile_data.iterrows():
        row = row[1]
        faces = non_empty[non_empty.userId == row.userid]
        if faces.size == 0:
            continue
        # randomly choose the first row
        X.append(faces.iloc[0].drop(labels=id_col))

        # age_group_name = age_to_age_group(row.age)
        # age_category_id = age_group_to_category_id(age_group_name)
        log_age = math.log(row.age)
        y.append(log_age)
    models = [
        # LinearRegression(),
        # LinearRegression(),
        # ExtraTreesRegressor(n_estimators=50),
        # RandomForestRegressor(n_estimators=50),
        GradientBoostingRegressor(n_estimators=500)
        # VotingRegressor(
        #     estimators=[
        #         ('a', RandomForestRegressor(n_estimators=100)),
        #         ('b', ExtraTreesRegressor(n_estimators=100)),
        #         ('c', GradientBoostingRegressor(n_estimators=100))],
        #     weights=(1, 2, 1),
        #     # voting='hard',
        #     n_jobs=4,
        # ),
        # VotingRegressor(
        #     estimators=[
        #         ('a', ExtraTreesRegressor(n_estimators=25)),
        #         ('b', ExtraTreesRegressor(n_estimators=50)),
        #         ('c', ExtraTreesRegressor(n_estimators=75))],
        #     weights=(1, 1, 1),
        #     # voting='hard',
        #     n_jobs=4),
    ]
    # method_params = [{"estimators": [
    #     ('a', RandomForestClassifier(n_estimators=100)),
    #     ('b', ExtraTreesClassifier(n_estimators=100)),
    #     ('c', GradientBoostingClassifier(n_estimators=100))],
    #     "weights": (1, 1, 2),
    #     "voting": 'hard',
    #     "n_jobs": 4}]

    X = np.array(X)
    y = np.array(y)
    k_fold(X, y, models, normalize_splits, eval_regressor_model)


if __name__ == "__main__":
    k_fold_train()
