import argparse
from sklearn.decomposition import PCA
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

from scripts.age_regressor_train import prepare_age_training_data
from utils.data_processing import normalize_splits, parse_input
from utils.label_mapping import age_to_age_group, age_group_to_category_id
import pickle


def run_pca(out_path):
    # read data
    DATA_PATH = "../../new_data/Train/"
    X, y = parse_input(DATA_PATH, is_train=True, map_age_to_group=False)
    X = prepare_age_training_data(X)
    # methods = [ModeModel, QDA, LogisticRegression, LDA, SVC,DecisionTreeClassifier,KNeighborsClassifier]
    X = np.array(X)
    pca = PCA(n_components=10)
    pca.fit(X)
    pickle.dump(pca, open(out_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./out.pca",
                        help='an integer for the accumulator')
    args = parser.parse_args()
    run_pca(args.output_path)
