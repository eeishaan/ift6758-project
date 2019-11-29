from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

import numpy as np


def age_score(ypred, ytest):
    acc = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())

    return acc


def gender_score(ypred, ytest):
    return accuracy_score(ytest, ypred)

def personality_score(ypred, ytest):
    return mean_squared_error(ytest, ypred)