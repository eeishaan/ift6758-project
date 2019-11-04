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

from util.data_processing import normalize_splits
from util.label_mapping import age_to_age_group, age_group_to_category_id
import pickle


def run_pca(out_path):
    # read data
    DATA_PATH = "../../new_data/Train/"
    image_data = pd.read_csv(os.path.join(DATA_PATH, "Image", "oxford.csv"))
    text_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "liwc.csv"))

    input_data = image_data.join(text_data.set_index('userId'), on="userId")
    # input_data = text_data

    profile_data = pd.read_csv(os.path.join(DATA_PATH, "Profile", "Profile.csv"))

    filtered_images = input_data[input_data.userId.isin(profile_data.userid)]

    non_empty = filtered_images.dropna()

    summary = non_empty.describe()
    # iterate and find columns where min, max and mean are all same
    irrelevant_columns = []
    for c in summary.columns:
        col_stats = summary[c]
        if col_stats['min'] == col_stats['mean'] or col_stats['max'] == col_stats['mean']:
            irrelevant_columns.append(c)

    non_empty.drop(labels=irrelevant_columns, axis=1, inplace=True)
    X = []
    id_col = ['userId', 'faceID']
    # id_col = ['userId']
    for row in profile_data.iterrows():
        row = row[1]
        faces = non_empty[non_empty.userId == row.userid]
        if faces.size == 0:
            continue
        # randomly choose the first row
        X.append(faces.iloc[0].drop(labels=id_col))
    # methods = [ModeModel, QDA, LogisticRegression, LDA, SVC,DecisionTreeClassifier,KNeighborsClassifier]
    X = np.array(X)
    pca = PCA(n_components=5)
    pca.fit(X)
    pickle.dump(pca, open(out_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./out.pca",
                        help='an integer for the accumulator')
    args = parser.parse_args()
    run_pca(args.output_path)
