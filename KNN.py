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
    X_train = X[:0.7 * int(len(X)), :]
    X_test = X[0.7 * int(len(X)):, :]
    y_train = y[:0.7 * int(len(X)), :]
    y_test = y[0.7 * int(len(X)):, :]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)