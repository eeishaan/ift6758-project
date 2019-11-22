from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

from utils.label_mapping import age_to_age_group
import numpy as np


def age_score(ypred, ytest):
    age_to_group_map = np.vectorize(age_to_age_group)

    acc = accuracy_score(age_to_group_map(ytest), ypred)
    cm = confusion_matrix(age_to_group_map(ytest), ypred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())

    return acc


def gender_score(ypred, ytest):
    return accuracy_score(ytest, ypred)


def personality_score(ypred, ytest):
    return mean_squared_error(ytest, ypred)
