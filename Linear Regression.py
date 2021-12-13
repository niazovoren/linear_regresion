import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Build arrays of random points

random_Num = np.random.randint(0, 100, 10)
# print(random_Num)

x = np.random.random(10)
# print(x)

g = np.random.randint(0, 10, 5) * 3
# print(g)

fibonacci_seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

random_fibonacci = np.random.choice(fibonacci_seq, 1)
# print(random_fibonacci)

# Preparation of labels for first Dataset

y_line_1 = g[1] * x + np.random.normal(10)

print(g[1])

# Preparation of labels for second Dataset

y_line_2 = g[2] * x + 5 + np.random.normal(10)
# print(y_line_2)

# Preparation of labels for second Dataset

y_line_3 = g[3] * (x ** 2) + 4 * x + 10 + np.random.normal(10)

# print(y_line_3)

# plt.figure()

# plt.subplot(311)
# plt.scatter(x, y_line_1)

# plt.subplot(312)
# plt.scatter(x, y_line_2)

# plt.subplot(313)
# plt.scatter(x, y_line_3)

# plt.show()

# Regression calculation for first Dataset
X_transpose = np.transpose(x.reshape((-1, 1)))
# print(np.shape(x))
# print(np.shape(X_transpose))
# print(X_transpose)
print((X_transpose @ x.reshape((-1, 1)))**-1 @ X_transpose)
h = ((X_transpose @ x.reshape((-1, 1)))**-1 @ X_transpose) @ y_line_1.reshape((-1, 1))
print(h)

plt.figure()
plt.plot(x, x * h)
plt.scatter(x, y_line_1)
plt.show()

# using linear regression with sklearn

reg = linear_model.LinearRegression()
h2 = reg.fit(x.reshape((-1, 1)), y_line_1)
print(reg.coef_)
