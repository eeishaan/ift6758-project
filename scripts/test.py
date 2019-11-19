import pandas as pd
import pickle

import joblib
from utils.data_processing import parse_input

# input_path = "/home/mila/teaching/user17/new_data/Train"
# X, y = parse_input(input_path, is_train=True, map_age_to_group=True)
#
# relational_data = X["relation"]
# relational_data=relational_data.drop(columns=[relational_data.columns[0]])
#
# i = relational_data.userid
# categories = list(set(relational_data["like_id"]))
# j = pd.Categorical(relational_data.like_id, categories=categories)
#
# test=pd.crosstab(i, j).astype(bool)
test = joblib.load(open("/home/mila/teaching/user17/processed_data/Xtrain_rel.pkl", mode="rb"))

# print(relational_data.columns)
# test = relational_data.pivot(index='userid', columns='like_id')
print(test[0:10])
