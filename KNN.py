import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def split_data():
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    np.random.seed(2)
    np.random.shuffle(X)
    np.random.seed(2)
    np.random.shuffle(y)
    X_train = X[:int(0.7 * len(X)), :]
    X_test = X[int(0.7 * len(X)):, :]
    y_train = y[:int(0.7 * len(X))]
    y_test = y[int(0.7 * len(X)):]
    return X_train, X_test, y_train, y_test


# X_train1, X_test1, y_train1, y_test1 = split_data()
# print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


