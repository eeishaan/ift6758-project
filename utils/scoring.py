import math

from sklearn.metrics import confusion_matrix

import numpy as np

from utils.label_mappings import age_to_age_group, category_id_to_age
from sklearn.metrics import accuracy_score, mean_squared_error

age_id_to_age_group_func = np.vectorize(category_id_to_age)


def age_score(ypred, ytest, age_to_group):
    """
    Calculates the age precision scores
    :param ypred: list of age predictions (age value or categories)
    :param ytest: list of age category labels
    :param age_to_group:  if True, it will convert the age value to its group
    :return: the accuracy score
    """
    if age_to_group is False:
        age_to_age_group_func = np.vectorize(age_to_age_group)
        ypred = age_to_age_group_func(ypred)
    else:
        ytest = age_id_to_age_group_func(ytest)
    acc = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())

    return acc


def gender_score(ypred, ytest):
    """
    Computes the gender precision score
    :param ypred: list of gender predictions
    :param ytest: list of gender lagels
    :return: the accuracy score
    """
    return accuracy_score(ytest, ypred)


def personality_score(ypred, ytest):
    """
    Computes the personality RMSE score
    :param ypred: list of a personality trait predictions
    :param ytest: list of a personality trait labels
    :return: the RMSE score
    """
    return math.sqrt(mean_squared_error(ytest, ypred))
