AGE_CATEGORIES = ["xx-24", "25-34", "35-49", "50-xx"]


def age_to_age_group(age):
    if age <= 24:
        age_group = AGE_CATEGORIES[0]
    elif age <= 34:
        age_group = AGE_CATEGORIES[1]
    elif age <= 49:
        age_group = AGE_CATEGORIES[2]
    elif age >= 50:
        age_group = AGE_CATEGORIES[3]
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


def age_group_to_category_id(age_group):
    return AGE_CATEGORIES.index(age_group)


def category_id_to_age(age_category_id):
    return AGE_CATEGORIES[age_category_id]
