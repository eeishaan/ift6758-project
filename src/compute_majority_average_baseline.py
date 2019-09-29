import pandas as pd
import argparse
from collections import Counter
from utils.dataset_utils import age_to_age_group
import json

from utils.dataset_utils import gender_id_to_name


def compute_most_frequent(labels):
    occurence_count = Counter(labels)
    most_common = occurence_count.most_common(1)[0][0]
    return most_common


def compute_mean(labels):
    total = 0
    for element in labels:
        total += element
    mean_value = total / len(labels)
    return mean_value


def compute_majority_average_baseline(profile_data_path, output_results_path=None):
    profile_data = pd.read_csv(profile_data_path)

    gender_names = [gender_id_to_name(gender_id) for gender_id in profile_data["gender"]]
    most_frequent_gender = compute_most_frequent(gender_names)
    print("Most frequent gender is {}".format(most_frequent_gender))

    age_groups = [age_to_age_group(age) for age in profile_data["age"]]
    mean_age = compute_most_frequent(age_groups)
    print("Most frequent age group is {}".format(mean_age))

    mean_ope = compute_mean(profile_data["ope"])
    print("Mean open is {}".format(mean_ope))

    mean_con = compute_mean(profile_data["con"])
    print("Mean conscientious is {}".format(mean_con))

    mean_ext = compute_mean(profile_data["ext"])
    print("Mean extrovert is {}".format(mean_ext))

    mean_agr = compute_mean(profile_data["agr"])
    print("Mean agreeable is {}".format(mean_agr))

    mean_neu = compute_mean(profile_data["neu"])
    print("Mean neurotic is {}".format(mean_neu))
    output_json = {
        "gender": most_frequent_gender,
        "age_group": mean_age,
        "open": str(mean_ope),
        "conscientious": str(mean_con),
        "extrovert": str(mean_ext),
        "agreeable": str(mean_agr),
        "neurotic": str(mean_neu)
    }
    if output_results_path is not None:
        with open(output_results_path, mode="w") as f:
            json.dump(output_json, f)
    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile_data_path', type=str, default=None,
                        help='Profile data path')
    parser.add_argument('--output_results_path', type=str, default=None,
                        help='Output json results path')
    args = parser.parse_args()
    compute_majority_average_baseline(args.profile_data_path, args.output_results_path)
