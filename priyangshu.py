# rk_methods.py

import numpy as np

def rk2(f, y0, t):
    y = np.array([y0])

    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = h * f(y[-1], t[i - 1])
        k2 = h * f(y[-1] + k1, t[i - 1] + h)

        y_next = y[-1] + 0.5 * (k1 + k2)
        y = np.append(y, [y_next], axis=0)

    return y

def rk4(f, y0, t):
    y = np.array([y0])

    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = h * f(y[-1], t[i - 1])
        k2 = h * f(y[-1] + 0.5 * k1, t[i - 1] + 0.5 * h)
        k3 = h * f(y[-1] + 0.5 * k2, t[i - 1] + 0.5 * h)
        k4 = h * f(y[-1] + k3, t[i - 1] + h)

        y_next = y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y = np.append(y, [y_next], axis=0)

    return y
