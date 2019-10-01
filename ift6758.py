#!/usr/bin/env python
import json
import os

import pandas as pd
import argparse
import xml.etree.ElementTree as ET

majority_average_dataset_info_path \
    = "mappings/majority_average_dataset_info.json"

TEAM_NAME = "user17"


def evaluate(test_data_dir, results_output_dir):
    os.makedirs(results_output_dir, exist_ok=True)


    with open(majority_average_dataset_info_path, "r") as f:
        majority_average_dataset_info = json.load(f)

    test_data = pd.read_csv(os.path.join(test_data_dir, "Profile/Profile.csv"))
    for index, user_data in test_data.iterrows():
        majority_average_dataset_info["id"] = str(user_data["userid"])
        users_root = ET.Element('user', attrib=majority_average_dataset_info)
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
