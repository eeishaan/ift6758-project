import pickle

import collections
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, \
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight, resample

import joblib
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC
from imblearn.under_sampling import ClusterCentroids
from utils.data_processing import parse_input, preprocess
from utils.label_mapping import age_to_age_group
import pandas as pd
import networkx as nx


def eval_classification_model(model, X_train, y_train, X_test, y_test):
    # evaluate train perf
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_cm = confusion_matrix(y_train, train_pred)
    train_cm = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]

    # test performance
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    test_cm = confusion_matrix(y_test, test_pred)
    test_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    return {
        'model': model,
        'train_acc': train_acc,
        "train_acc_per_class": train_cm.diagonal(),
        'test_acc': test_acc,
        "test_acc_per_class": test_cm.diagonal()
    }


def eval_regressor_model(model, X_train, y_train, X_test, y_test, is_log_age=True):
    # fit model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    if is_log_age:
        y_train = np.exp(y_train)
        y_test = np.exp(y_test)
        train_pred = np.exp(train_pred)
        test_pred = np.exp(test_pred)

    age_to_group_map = np.vectorize(age_to_age_group)
    # evaluate train perf
    train_acc = accuracy_score(age_to_group_map(y_train), age_to_group_map(train_pred))
    train_cm = confusion_matrix(age_to_group_map(y_train), age_to_group_map(train_pred))
    train_cm = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]

    # test performance
    test_acc = accuracy_score(age_to_group_map(y_test), age_to_group_map(test_pred))
    test_cm = confusion_matrix(age_to_group_map(y_test), age_to_group_map(test_pred))
    test_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    return {
        'model': model,
        'train_acc': train_acc,
        "train_acc_per_class": train_cm.diagonal(),
        'test_acc': test_acc,
        "test_acc_per_class": test_cm.diagonal()
    }


def split_train_val_test(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (X_train, y_train), (X_test, y_test)


def normalize_data(data):
    X_std, standard_scalers = preprocess(data, StandardScaler)
    # normalize train
    X_norm, norm_scalers = preprocess(X_std, MinMaxScaler)
    return X_norm


def prepare_age_training_data(data):
    image_data = data["image"]
    data["nrc"].rename(columns={'anger': 'anger_1'},
                       inplace=True)
    input_data = data["nrc"]

    input_data = input_data.join(data["liwc"].set_index('userId'), on="userId")
    # input_data = input_data.join(image_data.set_index('userId'), on="userId")
    # input_data = input_data.join(relational_data.set_index('userid'), on="userId")
    # input_data = joblib.load(open("/home/mila/teaching/user17/processed_data/Xtrain_rel.pkl", mode="rb"))
    relational_data = joblib.load(open("/home/mila/teaching/user17/processed_data/Xtrain_rel.pkl", mode="rb"))
    # for i in range(relational_data.shape[1]):
    #     input_data[str(i)] = relational_data[:, i]
    input_data = input_data.drop(columns=["userId"])
    # input_data = input_data.drop(columns=["faceID"])
    input_data = input_data.fillna(0)
    input_data = normalize_data(input_data)

    # summary = input_data.describe()
    # # iterate and find columns where min, max and mean are all same
    # irrelevant_columns = []
    # for c in summary.columns:
    #     col_stats = summary[c]
    #     if col_stats['min'] == col_stats['mean'] or col_stats['max'] == col_stats['mean']:
    #         irrelevant_columns.append(c)
    # print(f"irrelevant columns : {irrelevant_columns}")
    # input_data.drop(labels=irrelevant_columns, axis=1, inplace=True)

    return input_data


def prepare_age_label_data(label, log_age_transform=True):
    if log_age_transform:
        label = np.log(label)
    return label


def regression(input_path, test_size):
    log_age = True
    X, y = parse_input(input_path, is_train=True, map_age_to_group=False)
    hidden_layer_sizes = [1000 for _ in range(10)]
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, batch_size=100, verbose=True, learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        solver="adam",
                        # n_iter_no_change=100
                        )
    models = [
        # MLPRegressor(hidden_layer_sizes=1000, verbose=True, learning_rate_init=0.00001, max_iter=1000),
        # GradientBoostingRegressor(n_estimators=100),
        mlp
    ]
    X_merged = prepare_age_training_data(X)

    age_labels = prepare_age_label_data(y["age"], log_age_transform=log_age)

    (X_train, y_train), (X_test, y_test) = split_train_val_test(X_merged, age_labels, test_size=test_size)
    X_train_resampled = X_train
    y_train_resampled = y_train
    print(f"frequency before resampling : {collections.Counter(y_train_resampled)}")
    # X_train_resampled, y_train_resampled =  SMOTENC(categorical_features=[30,40,50]).fit_resample(X_train, y_train)
    print(f"frequency after resampling : {collections.Counter(y_train_resampled)}")
    # sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight = None
    for model in models:
        model.fit(X_train_resampled, y_train_resampled)
        results = eval_regressor_model(model, X_train, y_train, X_test, y_test, is_log_age=log_age)
        print(results)


def run_pca(X, n_components=3):
    X = np.array(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print(pca.explained_variance_.round(2))
    return pca


def classification(input_path, test_size):
    X, y = parse_input(input_path, is_train=True, map_age_to_group=True)
    from itertools import groupby
    X_merged = prepare_age_training_data(X)
    pca = pickle.load(open("/home/mila/teaching/user17/philippe_lelievre/scripts/out.pca", "rb"))
    # pca = run_pca(X_merged, n_components=3)
    # X_merged = pca.transform(X_merged)

    hidden_layer_sizes = [1000 for _ in range(100)]
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, batch_size=100, verbose=True, learning_rate_init=0.001,
                        max_iter=500,
                        # early_stopping=True,
                        solver="adam",
                        # n_iter_no_change=100
                        )
    models = [
        # KNeighborsClassifier(10),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        # mlp,

        # RandomForestClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        # GradientBoostingClassifier(n_estimators=50),
        # GaussianProcessClassifier(),
        # RBF(),
        # KNeighborsClassifier(),
        # RidgeClassifier(),
        # GradientBoostingClassifier(n_estimators=100),
        VotingClassifier(
            estimators=[
                ('a', RandomForestClassifier(n_estimators=10)),
                ('b', ExtraTreesClassifier(n_estimators=10)),
                ('c', GradientBoostingClassifier(n_estimators=100)),

            ],
            weights=(1, 1, 1),
            voting='hard',
            n_jobs=4
        )
    ]
    # for i, model in enumerate(models):
    #     models[i] = BalancedBaggingClassifier(n_estimators=1, base_estimator=model,
    #                                           sampling_strategy='auto',
    #                                           replacement=False,
    #                                           random_state=0),

    age_labels = y["age"]

    (X_train, y_train), (X_test, y_test) = split_train_val_test(X_merged, age_labels, test_size=test_size)
    # sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    X_train_resampled = X_train
    y_train_resampled = y_train
    print(f"frequency before resampling : {collections.Counter(y_train_resampled)}")
    X_train_resampled, y_train_resampled =  ADASYN().fit_resample(X_train, y_train)
    print(f"frequency after resampling : {collections.Counter(y_train_resampled)}")
    # oversampling_sample_weight = compute_sample_weight(class_weight="balanced", y=y_resampled)

    for model in models:
        model = model
        try:
            model.fit(X_train_resampled, y_train_resampled)
            results = eval_classification_model(model, X_train, y_train, X_test, y_test)
            print(results)
        except AttributeError:
            # Method does not exist; What now?
            print("Failed...")


def train():
    input_path = "/home/mila/teaching/user17/new_data/Train"
    mode = "classification"
    test_size = 0.2
    if mode == "regression":
        regression(input_path, test_size)
    elif mode == "classification":
        classification(input_path, test_size)
    else:
        raise Exception("Wrong training mode")


if __name__ == "__main__":
    train()
