import numpy as np


class solution():
    
    def __init__(self, tsol, ysol):     
        self.t = tsol
        self.y = ysol

def RK45_step(fun, t_init, y_init, step, args):
    "Step of Runge-Kutta"

    k1 = np.array(fun(t_init, y_init, *args))
    k2 = np.array(fun(t_init + step/2, y_init + k1 * step/2, *args))
    k3 = np.array(fun(t_init + step/2, y_init + k2 * step/2, *args))
    k4 = np.array(fun(t_init + step, y_init + k3 * step, *args))

    return t_init + step, y_init + (k1 + 2*k2 + 2*k3 + k4) * step/6

def RK45_vdp(fun, y0, t0, tf, step, eps, a_function):
    "Manual Runge-Kutta of order 4(5) for the modified van der Pol with changing a."

    
    tlist = [t0]
    ylist = [y0]
    
    t_it = t0
    y_it = y0
    while t_it < tf:

        args = (eps, a_function(t_it))
        t_it, y_it = RK45_step(fun, t_it, y_it, step, args=args)
        
        tlist.append(t_it)
        ylist.append(y_it)
        
    return solution(np.array(tlist), np.array(ylist).T)

###############################

def RK45_Hopf(fun, y0, t0, tf, step, w, mu_function):
    "Manual Runge-Kutta of order 4(5) for the Hopf with changing mu."
    
    tlist = [t0]
    ylist = [y0]
    
    t_it = t0
    y_it = y0
    while t_it < tf:

        args = (w, mu_function(t_it))
        t_it, y_it = RK45_step(fun, t_it, y_it, step, args=args)
        
        tlist.append(t_it)
        ylist.append(y_it)
        
    return solution(np.array(tlist), np.array(ylist).T)
    
###############################

def RK45_SNILC(fun, y0, t0, tf, step, mu, w_function):
    "Manual Runge-Kutta of order 4(5) for the SNILC with changing mu."
    
    tlist = [t0]
    ylist = [y0]
    
    t_it = t0
    y_it = y0
    while t_it < tf:

        args = (w_function(t_it), mu)
        t_it, y_it = RK45_step(fun, t_it, y_it, step, args=args)
        
        tlist.append(t_it)
        ylist.append(y_it)
        
    return solution(np.array(tlist), np.array(ylist).T)
