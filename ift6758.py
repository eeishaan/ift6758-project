#!/usr/bin/env python
import json
import os
from sklearn.externals import joblib
from utils.data_processing import parse_output

import pandas as pd
import argparse
import xml.etree.ElementTree as ET

model_path = "trained_models/baseline.pkl"
TEAM_NAME = "user17"


def evaluate(test_data_dir, results_output_dir):
    os.makedirs(results_output_dir, exist_ok=True)
    model = joblib.load(model_path)

    test_data = pd.read_csv(os.path.join(test_data_dir, "Profile/Profile.csv"))
    pred_df = model.predict(test_data)
    pred_df = parse_output(pred_df)

    for index, user_data in test_data.iterrows():
        pred = {}
        pred["id"] = str(user_data["userid"])
        pred.update(dict(pred_df.loc[user_data['userid']]))
        users_root = ET.Element('user', attrib=pred)
        xml_string_data = ET.tostring(users_root, encoding="unicode")
        xml_file = open(os.path.join(results_output_dir, "{}.xml".format(user_data["userid"])), "w")
        xml_file.write(xml_string_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default=None,
                        help='Data input directory')
    parser.add_argument('-o', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()
    evaluate(args.i, args.o)
