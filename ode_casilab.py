# Solve ode using RK4

import numpy as np


def solve_rk4(ode, t, y0):
    """ ode: equation of order one as function of the form ode(t, y0)
        t: points at which the values are to be evaluated (as array)
        y0: initial value 

        returns the solution as an array 
            
        y = solve_rk4(ode, t, y0)

        plot(t,y)
    """
    y = np.zeros(len(t))
    t0 = t[0]
    h = (t[-1] - t[0])/len(t)
    for i in range(len(t)):
        k1 = h * ode(t0 , y0)
        k2 = h*ode(t0 + h/2, y0 + k1/2)
        k3 = h*ode(t0 + h/2, y0 + k2/2)
        k4 = h*ode(t0 + h, y0 + k3)
        
        y[i] = y0
        t0 += h
        y0 = y0 + (k1+2*(k2+k3)+k4)/6

    return y