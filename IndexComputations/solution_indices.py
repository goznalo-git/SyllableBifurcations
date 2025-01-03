"""
Title: Calculation of spectral and envelope indices for the different bifurcations
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import argrelextrema, find_peaks
import scipy

import os
import sys
sys.path.append("../")

from syllable_indices import *
from dynamical_systems import airflow


##################
### PARAMETERS ###
##################

distance=200
height_min=0.002


#########################
### INDEX COMPUTATION ###
#########################

indices_VdP = dict()
indices_Hopf = dict()
indices_SNILC = dict()

folder = "Solutions_subsampled"
filelist = os.listdir(folder)
filelist.sort()
for file in filelist:

    print("Processing file", file)

    if ".pkl" not in file:
        continue
        
    #################
    ### OPEN FILE ###
    #################
    
    with open(f"{folder}/{file}", 'rb') as f:
        soldict = pickle.load(f)
        tsol = soldict["tsol"]
        xsol = soldict["xsol"]
        lam1sol = soldict["lambda1"]

    vsol = np.diff(xsol)
    vsol = np.append(vsol, 0)
    
    dt = tsol[1] - tsol[0]

    n = int(file.split(".")[0].split("_")[1])
    
    #########################
    ### SPECTRAL ANALYSIS ###
    #########################

    try:
        # Compute the FFT
        xf = fftfreq(len(xsol), dt)[:len(xsol)//2]
        yf = fft(xsol)
        spectrum = np.abs(yf)[:len(xsol)//2]
    except ValueError:
        print("###",file,"###")
        print("Simulation with parameter lambda1 = ",lam1sol, "fails to Fourier transform")
        continue
        
    # Keep only the peaks which are separated and high enough
    freqs, heights = spectrum_peaks(xf, spectrum[:100000], height_min=height_min, distance=distance)

    # Obtain the indices
    center, roughness, iIRR = measures(freqs, heights)

    
    ##############################
    ### ENVELOPE INTERPOLATION ###
    ##############################
        
    if "VdP" in file:
        
        # Find peaks twice
        tp = tsol[scipy.signal.find_peaks(vsol,prominence=0.0001)[0]]
        vp = vsol[scipy.signal.find_peaks(vsol,prominence=0.0001)[0]]

        tpp = tp[scipy.signal.find_peaks(vp)[0]]
        vpp = vp[scipy.signal.find_peaks(vp)[0]]

        # Get datapoints before and after oscillations
        first_index = np.where(tsol == tpp[0])[0][0]
        last_index = np.where(tsol == tpp[-1])[0][0]
        
        # Add an datapoints before and after oscillations
        tpp = np.append(tsol[:first_index], tpp)
        vpp = np.append(vsol[:first_index], vpp)
        
        tp = np.append(tpp, tsol[last_index:])
        vp = np.append(vpp, vsol[last_index:])


    elif "Hopf" in file or "SNILC" in file:
        
        # Find peaks
        tp = tsol[scipy.signal.find_peaks(vsol)[0]]
        vp = vsol[scipy.signal.find_peaks(vsol)[0]]
    
    else: # other files
        raise Exception("File not recognized as VdP/Hopf/SNILC. Exiting.")

    
    # Interpolate points:
    x_interp = np.interp(tsol, tp, np.abs(vp))

    
    ####################
    ### FIT ENVELOPE ###
    ####################
    
    try:
        params, _ = scipy.optimize.curve_fit(airflow, tsol, x_interp, p0=None)
        lam1 = params[0]
        lam2 = params[1]
        if lam1 < 0 or lam2 < 0:
            print("This simulation (lambda1 = ",lam1sol, ") fits with negative lambdas")
            continue
            
    except RuntimeError:
        print("This simulation (lambda1 = ",lam1sol, ") fails to fit")
        continue
        

    #######################################
    ### SAVE INDICES IF EVERYTHING OKAY ###
    #######################################
    
    if "VdP" in file:
        indices_VdP[n] = {"center":center, "roughness":roughness, 
                          "iIRR":iIRR, "lam1":lam1, "lam2":lam2}
    elif "Hopf" in file:
        indices_Hopf[n] = {"center":center, "roughness":roughness,
                           "iIRR":iIRR, "lam1":lam1, "lam2":lam2}
    elif "SNILC" in file:
        indices_SNILC[n] = {"center":center, "roughness":roughness,
                            "iIRR":iIRR, "lam1":lam1, "lam2":lam2}


###############################
### SAVE FILES WITH INDICES ###
###############################

with open(f"Indices/subsampled_VdP.pkl", 'wb') as f:
    pickle.dump(indices_VdP, f)
    
with open(f"Indices/subsampled_Hopf.pkl", 'wb') as f:
    pickle.dump(indices_Hopf, f)

with open(f"Indices/subsampled_SNILC.pkl", 'wb') as f:
    pickle.dump(indices_SNILC, f)

















