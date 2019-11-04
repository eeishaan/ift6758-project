import os

from util.data_processing import remove_empty_data
from util.label_mapping import age_to_age_group, age_group_to_category_id
import pandas as pd

DATA_PATH = "../../new_data/Train/"


def visualize_age():
    image_data = pd.read_csv(os.path.join(DATA_PATH, "Image", "oxford.csv"))
    liwc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "liwc.csv"))
    nrc_data = pd.read_csv(os.path.join(DATA_PATH, "Text", "nrc.csv"))
    nrc_data.rename(columns={'anger': 'anger_1'},
                    inplace=True)
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
        age_category_id = age_group_to_category_id(age_group_name)
        y.append(age_category_id)




if __name__ == "__main__":
    visualize_age()
