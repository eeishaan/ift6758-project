from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def age_score(ypred, ytest):
    # There is a specific function here just in case we want add a
    # task specific label encoding or other transformation at some point.
    return accuracy_score(ytest, ypred)

def gender_score(ypred, ytest):
    return accuracy_score(ytest, ypred)

def personality_score(ypred, ytest):
    return np.sqrt(mean_squared_error(ytest, ypred))