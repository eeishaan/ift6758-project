import os

import pandas as pd
from sklearn.model_selection import train_test_split
from .label_mappings import *


def preprocess(df, scaler_type, scalers=None):
    '''
    df: DataFrame to be processed
    scaler_type: sklearn.preprocessing scaler class
    scalers: If supplied each column would be scaled
        with the provided scaler
    '''
    used_scalers = {}
    ret_df = df.copy(deep=True)
    for c in df.columns:
        if c in ['userId', 'faceID']:
            continue
        if scalers is None:
            c_scaler = scaler_type()
            used_scalers[c] = c_scaler
            func = c_scaler.fit_transform
        else:
            c_scaler = scalers[c]
            func = c_scaler.transform
        ret_df[c] = func(df[[c]].values.astype(float))
    if scalers is None:
        return ret_df, used_scalers
    return ret_df, scalers


def parse_text_data(root):
    text_data_liwc = pd.read_csv(os.path.join(root, "Text", "liwc.csv"))
    text_data_liwc.columns = text_data_liwc.columns.str.lower()
    text_data_nrc = pd.read_csv(os.path.join(root, "Text", "nrc.csv"))
    text_data_nrc.columns = text_data_nrc.columns.str.lower()

    # Merge feature sources and drop nans
    text_data = pd.merge(left=text_data_liwc, right=text_data_nrc, on='userid')
    text_data = text_data.rename(columns={"userid": "userId"})
    text_data = text_data.set_index("userId")
    text_data = text_data.sort_index()
    text_data.fillna(dict(text_data.mean(axis=0)), inplace=True)

    return text_data



def parse_relational_data(relation_file, profile_data):
    relation_data = pd.read_csv(relation_file)
    relation_data = relation_data.rename(columns={"userid": "userId"})
    relation_data = relation_data.set_index("userId")
    relation_data = relation_data.sort_index()
    return relation_data


def parse_image_data(image_file, profile_data, is_train):
    image_data = pd.read_csv(image_file)
    # don't filter for test data
    # TODO: Why not? ^
    if is_train:
        image_data = image_data[image_data.userId.isin(profile_data.index)]
        image_data = image_data.dropna()
    X = []
    y = []
    id_col = ['faceID']
    image_cols = image_data.columns
    for i, row in enumerate(profile_data.iterrows()):
        row = row[1]
        faces = image_data[image_data.userId == profile_data.index[i]]
        if faces.size == 0:
            # Add a row full of None so that we don't miss out some profiles
            row_data = pd.Series([profile_data.index[i]] + [None] * (len(image_cols) - 1), index=image_cols)
        else:
            # randomly choose the first row for train data when there are multiple faces
            row_data = faces.iloc[0]

        X.append(row_data)
        if is_train:
            y.append(row.gender)
    X = pd.DataFrame(X)
    X = X.set_index("userId")
    X = X.sort_index()
    X = X.drop(columns=id_col)
    if is_train:
        return (X, y)
    return (X,)


def parse_input(root, is_train=True, age_to_group=True):
    profile_data = pd.read_csv(os.path.join(
        root, 'Profile/Profile.csv')).drop('Unnamed: 0', axis=1)
    profile_data = profile_data.set_index("userid")
    profile_data = profile_data.sort_index()

    image_file = os.path.join(root, 'Image', 'oxford.csv')
    image_data = parse_image_data(image_file, profile_data, is_train)

    text_data = parse_text_data(root)

    relational_file = os.path.join(root, 'Relation', 'Relation.csv')
    relational_data = parse_relational_data(relational_file, profile_data)

    # Keep as dict and delegate to the submodels for dealing with different data sources
    X = {"user_id": profile_data.index, "image": image_data[0],
         "relation": relational_data, "text": text_data}

    if is_train:
        # need to explicitly construct this as we don't have equal number of
        # face data as profiles
        y = process_labels(profile_data, age_to_group=age_to_group)
        return X, y
    return X


def process_labels(profile_data, age_to_group=True):
    if age_to_group:
        profile_data['age'] = profile_data['age'].apply(lambda x: age_to_age_group(x))
        profile_data['age'] = profile_data['age'].apply(lambda x: age_group_to_category_id(x))
    profile_data.filter(items=['age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])
    return profile_data


def parse_output(pred_df):
    pred_df['gender'] = pred_df['gender'].apply(lambda x: gender_id_to_name(x))
    pred_df['age'] = pred_df['age']
    pred_df['ope'] = pred_df['ope'].apply(lambda x: str(x))
    pred_df['con'] = pred_df['con'].apply(lambda x: str(x))
    pred_df['ext'] = pred_df['ext'].apply(lambda x: str(x))
    pred_df['agr'] = pred_df['agr'].apply(lambda x: str(x))
    pred_df['neu'] = pred_df['neu'].apply(lambda x: str(x))
    pred_df = pred_df.rename(
        columns={
            "age": "age_group",
            "ope": "open",
            "con": "conscientious",
            "ext": "extrovert",
            "agr": "agreeable",
            "neu": "neurotic"
        })
    return pred_df


def split_data(X, y, split=0.2):
    train_ix, test_ix = train_test_split(X['user_id'], test_size=split)
    X_image_train, X_image_test = X['image'].loc[train_ix], X['image'].loc[test_ix]
    X_text_train, X_text_test = X['text'].loc[train_ix], X['text'].loc[test_ix]
    X_rel_train, X_rel_test = X['relation'].loc[train_ix], X['relation'].loc[test_ix]
    X_prof_train, X_prof_test = train_ix, test_ix
    ytrain, ytest = y.loc[train_ix], y.loc[test_ix]

    Xtrain = {"user_id": X_prof_train, "image": X_image_train,
              "relation": X_rel_train, "text": X_text_train}
    Xtest = {"user_id": X_prof_test, "image": X_image_test,
             "relation": X_rel_test, "text": X_text_test}

    return Xtrain, Xtest, ytrain, ytest
