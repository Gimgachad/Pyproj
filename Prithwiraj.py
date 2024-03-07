import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def megaode(f, x0, t, h, a, b, n):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

# def rk1(f, x0, t, h):
    n = len(t)
    x1 = np.zeros((n, len(x0)))
    x1[0] = x0
    for i in range(n - 1):
        x1[i+1] = x1[i] + h * f(x1[i],t[i])
    # if len(x0) >= 3:
    #     fig = plt.figure(figsize=(14,17))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x1[:, 0], x1[:, 1], x1[:, 2], lw=0.5)
    #     ax.set_xlabel("X-Axis")
    #     ax.set_ylabel("Y-Axis")
    #     ax.set_zlabel("Z-Axis")
    #     ax.set_title("Lorenz Attractor by RK4")
    #     plt.show()
    # else:
    #     plt.figure(figsize=(13, 7))
    #     for i in range(len(x0)):
    #         plt.plot(t, x1[:, i], label=f'x{i}(t)')
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('func')
    #     plt.title('RK1 Method')
    #     plt.grid(True)
    #     plt.show()
    
    # return x1
# def rk2(f, x0, t, h, ode, func, ):
    n = len(t)
    x2 = np.zeros((n, len(x0)))
    x2[0] = x0
    for i in range(n - 1):
        k1 = h * f(x2[i], t[i])
        k2 = h * f(x2[i] + k1/2, t[i] + h/2)
        x2[i+1] = x2[i] + 0.5 * (k1 + k2)
    # if len(x0) >= 3:
    #     fig = plt.figure(figsize=(14,17))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x2[:, 0], x2[:, 1], x2[:, 2], lw=0.5)
    #     ax.set_xlabel("X-Axis")
    #     ax.set_ylabel("Y-Axis")
    #     ax.set_zlabel("Z-Axis")
    #     ax.set_title("Lorenz Attractor by RK4")
    #     plt.show()
    # else:
    #     plt.figure(figsize=(13, 7))
    #     for i in range(len(x0)):
    #         plt.plot(t, x2[:, i], label=f'x{i}(t)')
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('func')
    #     plt.title('RK2 Method')
    #     plt.grid(True)
    #     plt.show()
    
    # return x2
# def rk3(f, x0, t, h):
    n = len(t)
    x3 = np.zeros((n, len(x0)))
    x3[0] = x0
    for i in range(n - 1):
        k1 = h * f(x3[i], t[i])
        k2 = h * f(x3[i] + k1/2, t[i] + h/2)
        k3 = h * f(x3[i] - k1 + 2 * k2, t[i] + h)
        x3[i+1] = x3[i] + (k1 + 4 * k2 + k3) / 6
    # if len(x0) >= 3:
    #     fig = plt.figure(figsize=(14,17))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x3[:, 0], x3[:, 1], x3[:, 2], lw=0.5)
    #     ax.set_xlabel("X-Axis")
    #     ax.set_ylabel("Y-Axis")
    #     ax.set_zlabel("Z-Axis")
    #     ax.set_title("Lorenz Attractor by RK4")
    #     plt.show()
    # else:
    #     plt.figure(figsize=(13, 7))
    #     for i in range(len(x0)):
    #         plt.plot(t, x3[:, i], label=f'x{i}(t)')
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('func')
    #     plt.title('RK3 Method')
    #     plt.grid(True)
    #     plt.show()
    
    # return x3
# def rk4(f, x0, t, h):
    """
    Solve a system of ODEs using the RK4 method.
    Parameters:
    f : function
        The function that defines the system of ODEs.
    x0 : numpy array
        The initial conditions.
    t : numpy array
        The time points where the solution should be computed.
    Returns:
    x : 2D numpy array
        The approximate solution at the time points in t.
    """
    n = len(t)
    x4 = np.zeros((n, len(x0)))
    x4[0] = x0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x4[i], t[i])
        k2 = h * f(x4[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(x4[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(x4[i] + k3, t[i] + h)
        x4[i+1] = x4[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    # if len(x0) >= 3:
    #     fig = plt.figure(figsize=(14,17))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x4[:, 0], x4[:, 1], x4[:, 2], lw=0.5)
    #     ax.set_xlabel("X-Axis")
    #     ax.set_ylabel("Y-Axis")
    #     ax.set_zlabel("Z-Axis")
    #     ax.set_title("Lorenz Attractor by RK4")
    #     plt.show()
    # else:
    #     plt.figure(figsize=(13, 7))
    #     for i in range(len(x0)):
    #         plt.plot(t, x4[:, i], label=f'x{i}(t)')
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('func')
    #     plt.title('RK4 method')
    #     plt.grid(True)
    #     plt.show()
    
    # return x4
    from sympy import Function, dsolve, Eq, Derivative, sin
    from sympy.abc import x
    # def solve_ode():
    #     # Define the function
    #     f = Function('f')
    #     # Define the differential equation
    #     # In this case, it's f'(x) - f(x) = sin(x)
    #     diff_eq = Eq(Derivative(f(x), x) - f(x), sin(x))
    #     # Solve the differential equation
    #     solution = dsolve(diff_eq)
    #     return solution
    #-------------OR-------------------------#
    # def solve_ode(ode, func):
    symsol = dsolve(ode, func)
        # return symsol
    # Example usage:
    # f = Function('f')
    # ode = Derivative(f(x), x, x) + 9*f(x)
    # print(solve_ode(ode, f(x)))
    from scipy.integrate import solve_ivp
    # def ivp(ode, t, x0):
    solivp = solve_ivp(ode, t, x0)
        # return solivp
    from scipy.integrate import odeint
    # def odent(ode, x0, t):
    solode = odeint(ode, x0, t)
        # return solode
    # def trapezoid(f, x0, t):
    n = len(t)
    x5 = np.zeros(n)
    x5[0] = x0
    for i in range(n - 1):
        x5[i+1] = x5[i] + h * (f(t[i], x5[i]) + f(t[i+1], x5[i] + h * f(t[i], x5[i]))) / 2
    # return x5
    import numpy as np
    # def simpsons_1_3_rule(f, a, b, n):
        # Calculate the step size
    h = (b - a) / n
    # Initialize the integration sum
    s = f(a) + f(b)
    # Calculate the sum in the formula
    for i in range(1, n):
        if i % 2 == 0:
            s += 2 * f(a + i * h)
        else:
            s += 4 * f(a + i * h)
    simsol = s * h / 3
    # Multiply by h/3
    # return simsol



# def plotall():
    if len(x0) >= 3:
        fig = plt.figure(figsize=(14,17))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2[:, 0], x2[:, 1], x2[:, 2], lw=0.5)
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_zlabel("Z-Axis")
        ax.set_title("Lorenz Attractor by RK4")
        plt.show()
    else:
        plt.figure(figsize=(13, 7))
        curves = [x1, x2, x3, x4, symsol, solivp, solode, x5, simsol] 
        colors = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'tab:pink', 'tab:purple'] 
        styles = ['-', '--', ':', '_.', '-.'] 
        for curve, color, style in zip(curves, colors, styles): 
            plt.plot(x, curve, c=color, linestyle=style) 
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('func')
        plt.title('All plots')
        plt.grid(True)
        plt.show()

    return plt.show()

    