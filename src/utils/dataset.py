def age_to_age_group(age):
    if age <= 24:
        age_group = "xx-24"
    elif age <= 34:
        age_group = "25-34"
    elif age <= 49:
        age_group = "35-49"
    elif age >= 50:
        age_group = "50-xx"
    else:
        age_group = "N/A"

    return age_group


def gender_id_to_name(gender_id):
    if gender_id == 0:
        gender_name = "male"
    elif gender_id <= 1:
        gender_name = "female"
    else:
        gender_name = "N/A"
    return gender_name
