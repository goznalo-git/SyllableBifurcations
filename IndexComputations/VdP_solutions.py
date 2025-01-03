"""
Title: (VdP bifurcation) Generation of solutions for different lambda1/lambda2

"""

import numpy as np
from collections import defaultdict
from itertools import product
from functools import partial
from copy import copy
import pickle

import sys
sys.path.append("../")

from dynamical_systems import Watery, airflow
from numerical_syllables import *
from aux_functions_manual import *


##############################
### INTEGRATION PARAMETERS ###
##############################

t0 = 0                  # Initial trajectory time
tf = 10                  # Final trajectory time 

subsample = 40
step = 1/(subsample * 44150.0)      # timestep
trange = np.arange(t0,tf,step)


###########################
### EQUATION PARAMETERS ###
###########################

eps = 0.0278  # CHANGE THIS TO MOVE THE SPECTRUM
y0 = (0,0)

# type = "slow"
type = "fast"

convolution = True # If true, multiply the parameter times the solution 

def a_function(t, height=-2*1e6/600, lambda1=3, lambda2=3, T1=7, T2=3):
    
    return 10*(height * airflow(t, lambda1, lambda2, T1, T2) + 0.01)

if type == "fast":
    a_function = partial(a_function, height=-4.5*1e15, lambda1=10, lambda2=10) # fast

#########################
### SOLUTION & SAVING ###
#########################

n = 0
for lam1 in np.linspace(9.5,10.5,50):

    print("Solution with lambda1 =", np.round(lam1,3), "lambda2 = 10")#3")
    
    sol = RK45_vdp(partial(Watery, prop=200000), y0, t0, tf, step, eps, partial(a_function, lambda1=lam1))
    
    tsol = sol.t[0::subsample]
    xsol = sol.y[0][0::subsample]
    ysol = sol.y[1][0::subsample]
    
    if convolution:
        aperture = np.array([a_function(t) for t in tsol])
        area = 1 - (aperture + 1)/2 # the aperture is what matters in VdP
        xsol = np.multiply(aperture,xsol)
    
    print("--Saving solution\n")
    with open(f"Solutions/VdP_{n:02}.pkl", 'wb') as f:
        pickle.dump({"tsol":tsol, "xsol":xsol, "lambda1":lam1}, f)

    n += 1


