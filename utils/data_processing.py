import os

import pandas as pd

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
    # TODO
    return None


def parse_relational_data(relation_file):
    # TODO
    return None


def parse_image_data(image_file, profile_data, is_train):
    image_data = pd.read_csv(image_file)
    # don't filter for test data
    if is_train:
        image_data = image_data[image_data.userId.isin(profile_data.userid)]
        image_data = image_data.dropna()

    X = []
    y = []
    id_col = ['userId', 'faceID']
    image_cols = image_data.columns
    for row in profile_data.iterrows():
        row = row[1]
        faces = image_data[image_data.userId == row.userid]

        # don't add to train data if features are unavailable
        if faces.size == 0 and is_train:
            continue

        if faces.size == 0:
            # Add a row full of None for test data so that we don't miss out some profiles
            row_data = pd.Series([None]*len(image_cols), index=image_cols)
        else:
            # randomly choose the first row for train data when there are multiple faces
            row_data = faces.iloc[0]

        X.append(row_data.drop(labels=id_col))
        if is_train:
            y.append(row.gender)
    X = pd.DataFrame(X)
    if is_train:
        return (X, y)
    return (X,)


def parse_input(root, is_train=True):
    profile_data = pd.read_csv(os.path.join(
        root, 'Profile/Profile.csv')).drop('Unnamed: 0', axis=1)

    image_file = os.path.join(root, 'Image', 'oxford.csv')
    image_data = parse_image_data(image_file, profile_data, is_train)

    text_root = parse_text_data(os.path.join(root, 'Text'))
    text_data = parse_text_data(text_root)

    relational_file = os.path.join(root, 'Relation', 'Relation.csv')
    relational_data = parse_relational_data(relational_file)

    # Keep as dict and delegate to the submodels for dealing with different data sources
    X = {"user_id": profile_data['userid'], "image": image_data[0],
         "relation": relational_data, "text": text_data}

    if is_train:
        # need to explicitly construct this as we don't have equal number of
        # face data as profiles
        y = {
            'age': profile_data['age'].apply(lambda x: age_to_age_group(x)),
            'gender': image_data[1],
            'ope': profile_data['ope'],
            'con': profile_data['con'],
            'ext': profile_data['ext'],
            'agr': profile_data['agr'],
            'neu': profile_data['neu'],
        }
        return X, y
    return X


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
