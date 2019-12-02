from sklearn.model_selection import KFold
import numpy as np
import math


def k_fold(X, y, model, age_to_group, n_splits):
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)
    per_task_results = {'gender': [],
                        'ope': [],
                        'con': [],
                        'ext': [],
                        'agr': [],
                        'neu': [],
                        'age': []}
    for train_index, test_index in kf.split(X['user_id']):
        X_train = {}
        X_test = {}
        train_user_ids = X['user_id'][train_index]
        test_user_ids = X['user_id'][test_index]
        for key, value in X.items():
            if key == "user_id":
                X_train[key], X_test[key] = train_user_ids, test_user_ids
            else:
                X_train[key], X_test[key] = X[key].loc[train_user_ids], X[key].loc[test_user_ids]
        y_train, y_test = y.loc[train_user_ids], y.loc[test_user_ids]
        model.fit(X_train, y_train)
        result = model.eval(X_test, y_test, age_to_group=age_to_group)
        for target_key, acc in result.items():
            per_task_results[target_key].append(acc)
    for key, result in per_task_results.items():
        print(
            f"{key} : {np.mean(result)} += {1.96 * np.std(result) / math.sqrt(len(result))}")
