from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.utils import compute_sample_weight


def k_fold(X, y, models, normalize_splits_func, eval_model_func):
    kf = KFold(n_splits=10)
    method_stats = {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index])
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = normalize_splits_func(X_train, X_test)
        # X_train = pca_model.transform(X_train)
        # X_test = pca_model.transform(X_test)
        sample_weight_train = sample_weight[train_index]
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        for m in models:

            ret = eval_model_func(m, train_data, test_data,sample_weight_train)
            m_name = type(m).__name__
            if m_name not in method_stats:
                method_stats[m_name] = []
            method_stats[m_name].append([ret['train_acc'], ret['test_acc']])
    for m_name, stats in method_stats.items():
        stats = np.array(stats)
        mean = np.mean(stats, axis=1)
        std = np.std(stats, axis=1)
        print('Method: {} Train Acc {}+={} Test Acc {}+={}'.format(m_name, mean[0], std[0], mean[1], std[1]))
