from typing import List, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Build arrays of random points

random_Num = np.random.randint(0, 100, 10)
# print(random_Num)

x = np.random.random(10)
x_reshaped = x.reshape(-1, 1)
# print(x)

g = np.random.randint(0, 10, 5) * 3
# print(g)

fibonacci_seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

random_fibonacci = np.random.choice(fibonacci_seq, 1)
# print(random_fibonacci)

# Preparation of labels for first Dataset

y_line_1 = (g[1] * x) + np.random.normal(10)

print(g[1])

# Preparation of labels for second Dataset

y_line_2 = (g[2] * x + 5) + np.random.normal(10)
# print(y_line_2)

# Preparation of labels for second Dataset

y_line_3 = g[3] * (x ** 2) + 4 * x + 10 + np.random.normal(10)

# Regression calculation for first Dataset

# print((X_transpose @ x.reshape((-1, 1)))**-1 @ X_transpose)
h = (((x @ x_reshaped) ** -1) * x) @ y_line_1.reshape((-1, 1))
print(h)

# plt.figure()
# plt.plot(x, x * h)
# plt.scatter(x, y_line_1)
# plt.show()

# using linear regression with sklearn

reg = linear_model.LinearRegression()
h2 = reg.fit(x.reshape((-1, 1)), y_line_1)
print(reg.coef_)

# Regression calculation for Second Dataset
print('\n \33[1;32m Second Data Results \33[1;0m')
print('The real a and b:', g[2], 5)
x_ones = np.ones(10)
x_ones = x_ones[:, np.newaxis]
x_new = np.hstack((x_reshaped, x_ones))
x_new_transpose = x_new.transpose()
result = ((np.linalg.inv(x_new_transpose @ x_new)) @ x_new_transpose @ y_line_2.reshape((-1, 1)))
print('Calculated a and b with normal equations:', result)

# Regression calculation for Second Dataset with sklearn

reg2 = linear_model.LinearRegression(fit_intercept=False)
result2 = reg2.fit(x_new, y_line_2)
print('Calculated a and b with sklearn model:', reg2.coef_)
'\n'

# Regression calculation for Third Dataset
print('\n \033[1;32m Third Data Results: \033[1;0m')
print('The real a, b and c:', g[3], 4, 10)
x_squared = x_reshaped ** 2
x_3 = np.hstack((x_squared, x_reshaped, x_ones))
x_3_transposed = x_3.transpose()
result3 = ((np.linalg.inv(x_3_transposed @ x_3)) @ x_3_transposed @ y_line_3.reshape((-1, 1)))
print('Calculated a, b and c with normal equations:', result3)

# Regression calculation for Second Dataset with sklearn

reg3 = linear_model.LinearRegression(fit_intercept=False)
result31 = reg3.fit(x_3, y_line_3)
print('Calculated a, b and c with sklearn model:', reg3.coef_)

# Show all data in figures
# plt.figure(1)
# plt.scatter(x, y_line_1)
# new_y = h * x
# plt.plot(np.linspace(0, 1, 10),np.linspace(0, 1, 10)*h, 'r')
# plt.title('First Data')
#
#
# plt.figure(2)
# plt.scatter(x, y_line_2)
# new_y_2 = result[0]*np.linspace(0, 1, 10) + result[1]
# plt.plot(np.linspace(0, 1, 10), new_y_2, 'r')
# plt.xlim([0, 1])
# plt.ylim([0, 33])
# plt.title('Second Data')
#
#
# plt.figure(3)
# plt.scatter(x, y_line_3)
# new_y_3 = result3[0]*(np.linspace(0, 1, 10))**2 + result3[1]*np.linspace(0, 1, 10) + result3[2]
# plt.plot(np.linspace(0, 1, 10), new_y_3, 'r')
# plt.xlim([0, 1])
# plt.ylim([0, 50])
# plt.title('Third Data')
# plt.show()

# Final question
x_last = [0.08750722, 0.01433097, 0.30701415, 0.35099786, 0.80772547, 0.16525226,
          0.46913072, 0.69021229, 0.84444625, 0.2393042, 0.37570761, 0.28601187,
          0.26468939, 0.54419358, 0.89099501, 0.9591165, 0.9496439, 0.82249202,
          0.99367066, 0.50628823]
y_last = [4.43317755, 4.05940367, 6.56546859, 7.26952699, 33.07774456, 4.98365345,
          9.93031648, 20.68259753, 38.74181668, 5.69809299, 7.72386118, 6.27084933,
          5.99607266, 12.46321171, 47.70487443, 65.70793999, 62.7767844, 35.22558438,
          77.84563303, 11.08106882]

x_last = np.array(x_last)
y_last = np.array(y_last)
ln_y = np.log(y_last)
y_reshaped = ln_y.reshape((-1,1))
print(ln_y)
x_squared_2 = (x_last.reshape((-1, 1)))**2
ones_column = np.ones((20, 1))
x_total = np.hstack((x_squared_2, x_last.reshape((-1, 1)), ones_column))
x_total_transpose = np.transpose(x_total)
final_h = np.linalg.inv(x_total_transpose @ x_total) @ x_total_transpose @ y_reshaped
print(final_h)

plt.figure(4)
plt.scatter(x_last, y_last)
exp_y = np.exp(final_h[2]) * np.exp(final_h[0]*(np.linspace(0, 1, 20))**2 + final_h[1] * np.linspace(0, 1, 20))
plt.plot(np.linspace(0, 1, 20), exp_y, 'r')
plt.title('final Data')
plt.show()




