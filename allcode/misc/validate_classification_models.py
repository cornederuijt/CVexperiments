import numpy as np
import numpy.random as rd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
import multiprocessing as mp
import math


class ClassificationModelValidator:
    # TODO: Some possible extensions:
    # TODO: over/under/SMOTE sampling
    # TODO: Alternative error metrics
    # TODO: Alternative validation schemes (cross validation, validation over time)
    # TODO: Include polynomial features?

    @staticmethod
    def run_auto_ml(X_train, X_valid, y_train, y_valid):
        # 1) Recombine but label the validation instances using PredefinedSplit
        # From documentation:
        # "For example, when using a validation set, set the test_fold to 0 for all samples that are part of the
        # validation set, and to -1 for all other samples."
        X_mat = np.vstack((X_train, X_valid))
        y_vec = np.hstack((y_train, y_valid))

        valid_indices = np.hstack((-1 * np.ones(y_train.shape[0]), np.zeros(y_valid.shape[0])))

        predef_valid_split = PredefinedSplit(valid_indices)

        # 1) Run various models and find proper hyper-parameters
        # Currently implemented:
            # 2) Random Forest
            # 3) XGBoost
            # 4) KNN classifier (commonly useful for explainability)

        # 1) Logistic regression (Typical benchmark)
        log_reg_result = ClassificationModelValidator.\
            train_hyper_param_logistic_regression(X_mat, y_vec, predef_valid_split)

    @staticmethod
    def train_hyper_param_logistic_regression(X_all, y_all,  predef_valid_split):
        params = {'C': np.array([1/100, 1/10, 1/2, 1, 2, 5, 10, 100])}
        model = LogisticRegression(tol=10**(-8), max_iter=10000)

        clf = GridSearchCV(model, params, cv=predef_valid_split).fit(X_all, y_all, n_jobs=mp.cpu_count()-1,
                                                                     return_train_score=True)
        return clf





