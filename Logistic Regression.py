import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Changing the values 2 to zero.
y[y != 0] = 1

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


def sigmond(z):
    return 1 / (1 + np.exp(-z))



