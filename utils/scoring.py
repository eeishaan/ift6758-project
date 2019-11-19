from sklearn.metrics import accuracy_score, mean_squared_error

def age_score(ypred, ytest):
    return None # TODO

def gender_score(ypred, ytest):
    return accuracy_score(ytest, ypred)

def personality_score(ypred, ytest):
    return mean_squared_error(ytest, ypred)