import os
import pandas as pd
from .label_mappings import *


def parse_text_data(param):
    # TODO
    return None


def parse_relational_data(param):
    # TODO
    return None


def parse_image_data(param):
    # TODO
    return None


def parse_input(input):
    image_data = parse_image_data(os.path.join(input, 'Image'))
    text_data = parse_text_data(os.path.join(input, 'Text'))
    relational_data = parse_relational_data(os.path.join(input, 'Relation'))
    profile_data = pd.read_csv(os.path.join(input, 'Profile/Profile.csv')).drop('Unnamed: 0', axis=1)

    # Keep as dict and delegate to the submodels for dealing with different data sources
    X = {"user_id": profile_data['userid'], "image": image_data, "relation": relational_data, "text": text_data}

    return X, profile_data


def parse_output(pred_df):
    pred_df['gender'] = pred_df['gender'].apply(lambda x: gender_id_to_name(x))
    pred_df['age'] = pred_df['age'].apply(lambda x: age_to_age_group(x))
    pred_df['ope'] = pred_df['ope'].apply(lambda x: str(x))
    pred_df['con'] = pred_df['con'].apply(lambda x: str(x))
    pred_df['ext'] = pred_df['ext'].apply(lambda x: str(x))
    pred_df['agr'] = pred_df['agr'].apply(lambda x: str(x))
    pred_df['neu'] = pred_df['neu'].apply(lambda x: str(x))
    return pred_df
