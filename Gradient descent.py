import numpy as np
import matplotlib.pyplot as plt
from Tools.scripts.nm2def import symbols
from sympy import symbols, diff, Symbol, Derivative

sum_loss_function = []
a = 2
b = 2
c = 0
x = [0, 1, 2]
x = np.array(x)
y = [1, 3, 7]
y = np.array(y)
# hypothesis class h(x) = a + b*x + c*x**2
f = 200
while f > 0:
    loss_function = sum((y - (a + b * x + c * (x ** 2))) ** 2)
    sum_loss_function.append(loss_function)

    dl_da = -2 * (y - (a + b * x + c * (x ** 2)))
    dl_db = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
    dl_dc = 2 * (y - (a + b * x + c * (x ** 2))) * (-2 * x * c)

    step_size_a = 0.1 * dl_da
    step_size_b = 0.1 * dl_db
    step_size_c = 0.1 * dl_dc

    a = a - step_size_a
    b = b - step_size_b
    c = c - step_size_c
    f = f - 1

print(len(sum_loss_function))
plt.plot(np.linspace(0, 200, 200), sum_loss_function)
plt.show()

while f > 0:
    loss_function = sum((y - (a + b * x + c * (x ** 2))) ** 2)
    sum_loss_function.append(loss_function)

    dl_da = -2 * (y - (a + b * x + c * (x ** 2)))
    dl_db = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
    dl_dc = 2 * (y - (a + b * x + c * (x ** 2))) * (-2 * x * c)

    step_size_a = 0.01 * dl_da
    step_size_b = 0.01 * dl_db
    step_size_c = 0.01 * dl_dc

    a = a - step_size_a
    b = b - step_size_b
    c = c - step_size_c
    f = f - 1

print(len(sum_loss_function))
plt.plot(np.linspace(0, 200, 200), sum_loss_function)
plt.show()

while f > 0:
    loss_function = sum((y - (a + b * x + c * (x ** 2))) ** 2)
    sum_loss_function.append(loss_function)

    dl_da = -2 * (y - (a + b * x + c * (x ** 2)))
    dl_db = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
    dl_dc = 2 * (y - (a + b * x + c * (x ** 2))) * (-2 * x * c)

    step_size_a = 1 * dl_da
    step_size_b = 1 * dl_db
    step_size_c = 1 * dl_dc

    a = a - step_size_a
    b = b - step_size_b
    c = c - step_size_c
    f = f - 1

print(len(sum_loss_function))
plt.plot(np.linspace(0, 200, 200), sum_loss_function)
plt.show()