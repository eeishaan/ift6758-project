AGE_CATEGORIES = ["xx-24", "25-34", "35-49", "50-xx"]


def age_to_age_group(age):
    """
    Converts a age float value to its category group name
    :param age: age value
    :return: age group
    """
    age = round(age)
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
    """
    Convert gender identifier to its gender name
    0 for male and 1 for female
    :param gender_id: gender identifier (0 or 1)
    :return: gender name
    """
    if gender_id == 0:
        gender_name = "male"
    elif gender_id <= 1:
        gender_name = "female"
    else:
        gender_name = "N/A"
    return gender_name


def age_group_to_category_id(age_group):
    """
    Converts a age group name to its category id
    :param age_group: Age group label
    :return: Category id
    """
    return AGE_CATEGORIES.index(age_group)


def category_id_to_age(age_category_id):
    """
    Converts a age group id to its corresponding group category name
    :param age_category_id: Age group id
    :return: Category group
    """
    return AGE_CATEGORIES[int(age_category_id)]
