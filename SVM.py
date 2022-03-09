import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load iris data and labels
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Spilt data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
y_train = np.where(y_train == 2, 1, y_train)
clf = SVC(C=1.0, kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test[:, :].reshape(-1, 2))
w = clf.coef_
# b = clf.intercept_
support_vectors = clf.support_vectors_
b = -clf.intercept_/w[0][1]
print(clf.coef_)
print(clf.intercept_)
a = -w[0][0]/w[0][1]
# print(a)
# print(b)
r = np.linspace(3, 8, 1000)
t = a*r + b
b_up_margin = -(clf.intercept_+1)/w[0][1]
b_down_margin = -(clf.intercept_-1)/w[0][1]
up_margin = a*r + b_up_margin
down_margin = a*r + b_down_margin
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.plot(r, t, color='k')
plt.plot(r, up_margin, color='r')
plt.plot(r, down_margin, color='r')
plt.show()
# print(y_predict)
# print(y_test)
# print((y_test != y_predict).sum()/len(y_test))

