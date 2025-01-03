"""
Title: (SNILC bifurcation) Generation of solutions for different lambda1/lambda2

"""

import numpy as np
from collections import defaultdict
from itertools import product
from functools import partial
from copy import copy
import pickle

import sys
sys.path.append("../")

from dynamical_systems import SNILC, airflow
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

mu = 1
y0 = (1,0)

# type = "slow"
type = "fast"

convolution = True # If true, multiply the parameter times the solution

def w_function(t, height=2.71*1e6/6, lambda1=3, lambda2=3, T1=7, T2=3):
    # Change the height of the function to move the spectrum
    return height * airflow(t, lambda1, lambda2, T1, T2) 

if type == "fast":
    w_function = partial(w_function, height=0.638*1e18, lambda1=10, lambda2=10) # fast


   
#########################
### SOLUTION & SAVING ###
#########################

n = 0
for lam1 in np.linspace(9.5,10.5,50):

    print("Solution with lambda1 =", np.round(lam1,3), "lambda2 = 10")#3")
    
    sol = RK45_SNILC(partial(SNILC, prop=5000), y0, t0, tf, step, mu, partial(w_function, lambda1=lam1))
    
    tsol = sol.t[0::subsample]
    rhosol = sol.y[0][0::subsample]
    phisol = sol.y[1][0::subsample]
    xsol = rhosol * np.cos(phisol)
    ysol = rhosol * np.sin(phisol)

    if convolution:
        gating = [w_function(t) for t in tsol]
        xsol = np.multiply(gating, xsol)
    
    print("--Saving solution\n")
    with open(f"Solutions/SNILC_{n:02}.pkl", 'wb') as f:
        pickle.dump({"tsol":tsol, "xsol":xsol, "lambda1":lam1}, f)

    n += 1


