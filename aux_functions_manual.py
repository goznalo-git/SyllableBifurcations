import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import defaultdict

from numerical_syllables import *


def plot_variables(F, y0, t0, tf, step, args, return_sol=False):
    """
    From an ODE dy/dt = F(y), with y vectorial and initial condition y0, solve the IVP
    from t0 to tf with step, and plot the resulting trajectory, as well as the individual variables x,y.
    Parameters in the function are specified as args=(a,b,c,...).
    """
    
    fig = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,2)
    ax = [ax0, ax1]
    
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.axis('equal')
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('x,y')

    sol = RK45(F, y0=y0, t0=t0, tf=tf, step=step, args=args)
    
    ax0.plot(sol.y[0], sol.y[1], alpha=0.5, color="brown")
    ax1.plot(sol.t, sol.y[0], label="x")
    ax1.plot(sol.t, sol.y[1], label="y")
    
    ax1.legend()
    
    if return_sol:
        return fig, ax, sol
    else:
        return fig, ax




def plot_nullclines(ax, k, delta, fixed_point=True):
    """
    Plot the nullclines in the specified axis,
    and optionally the fixed point.
    """
    
    x0 = -delta/k
    y0 = x0**3 / 3 - x0
    
    ax.vlines(x0, -2, 2, color="c", alpha=0.3, linestyle="--", label='Nullcline1: dy/dt=0')
    
    xs = np.linspace(-2, 2,1000)
    ax.plot(xs, xs**3 / 3 - xs, color="m", alpha=0.3, linestyle="--", label='Nullcline2: dx/dt=0')
    
    if fixed_point:
        print("Analytical fixed point:", (x0,y0))
        ax.scatter(x0, y0, marker="*", color="k", s=80, label="Fixed point")
        
    ax.legend()
    
    return ax
