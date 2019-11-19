import pickle

import operator
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.utils import compute_sample_weight

from scripts.age_regressor_train import split_train_val_test
from utils.data_processing import normalize_splits, remove_empty_data
from utils.k_fold import k_fold
from utils.label_mapping import age_to_age_group, age_group_to_category_id


def eval_model(model, train_data, test_data, sample_weight=None):
    # fit model
    model.fit(*train_data, sample_weight=sample_weight)

    # evaluate train perf
    train_pred = model.predict(train_data[0])
    train_acc = accuracy_score(train_data[1], train_pred)

    # test performance
    test_pred = model.predict(test_data[0])
    test_acc = accuracy_score(test_data[1], test_pred)

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


AGE_SAMPLE_WEIGHT = {
    'xx-24': 4244, '25-34': 1816, '35-49': 822, '50-xx': 292,
}


# AGE_ID_SAMPLE_WEIGHT = {
#     'xx-24': 4244, '25-34': 1816, '35-49': 822, '50-xx': 292,
# }
def eval(X, y, models, normalize_splits_func, eval_model_func):
    method_stats = {}
    (X_train, y_train), (X_test, y_test) = split_train_val_test(X, y, test_size=0.2)

    # X_train, X_test = normalize_splits_func(X_train, X_test)
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)
    for m in models:
        ret = eval_model_func(m, train_data, test_data)
        m_name = type(m).__name__
        if m_name not in method_stats:
            method_stats[m_name] = []
        method_stats[m_name].append([ret['train_acc'], ret['test_acc']])

    print(method_stats)


def k_fold_train():
    # read data
    DATA_PATH = "../../new_data/Train/"
    pca_model = pickle.load(open("out.pca", 'rb'))
    image_data = pd.read_csv(os.path.join(DATA_PATH, "Image", "oxford.csv"))
    liwc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "liwc.csv"))
    nrc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "nrc.csv"))
    nrc_data.rename(columns={'anger': 'anger_1'}, inplace=True)
    input_data = image_data.join(liwc_data.set_index('userId'), on="userId")
    input_data = input_data.join(nrc_data.set_index('userId'), on="userId")

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

        age_group_name = age_to_age_group(row.age)
        # age_category_id = age_group_to_category_id(age_group_name)
        y.append(age_group_name)

    # RandomForestClassifier(n_estimators=50)),
    # ('b', ExtraTreesClassifier(n_estimators=50)),
    # ('c', GradientBoostingClassifier(n_estimators=100))],
    # methods = [ModeModel, QDA, LogisticRegression, LDA, SVC, DecisionTreeClassifier, KNeighborsClassifier,
    #     #            DecisionTreeClassifier]
    # methods = [DecisionTreeClassifier]

    # methods = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
    # method_params = [{}, {}, {'solver': 'newton-cg'}, {}, {'gamma': 'auto'}]
    # method_params = [{"n_estimators": 100}, {"n_estimators": 100}, {"n_estimators": 100}]
    methods = [
        RandomForestClassifier(n_estimators=50),
        ExtraTreesClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=1000),
    #     VotingClassifier(
    #         estimators=[
    #             ('a', RandomForestClassifier(n_estimators=50)),
    #             ('b', ExtraTreesClassifier(n_estimators=50)),
    #             ('c', GradientBoostingClassifier(n_estimators=100))],
    #         weights=(1, 2, 1),
    #         voting='hard',
    #         n_jobs=4
    #     )
     ]
    X = np.array(X)
    y = np.array(y)
    eval(X, y, methods, normalize_splits, eval_model)


if __name__ == "__main__":
    k_fold_train()
