import numpy as np
import scipy

#### SPECTRAL INDICES ####

def spectrum_peaks(xf, spectrum, height_min=0.001, distance=200):
    """
    Given a Fourier spectrum, compute the frequencies and heights 
    associated to the several peaks, assuming distributed as 
    multiples of the fundamental (highest) peak.
    """

    # Find peaks with the standard scipy function (indices, freqs, heights)
    local_max = (scipy.signal.find_peaks(spectrum,
                                         height=np.max(spectrum)*height_min,
                                         distance=distance)[0],)
    xf_local_max = xf[local_max]
    spectrum_local_max = spectrum[local_max]

    # Find index, frequency and height of the highest (fundamental)
    if np.isclose(xf_local_max[0], 0, atol=1e0): #Ignore the initial peak (f0=0) in Hopf/SNILC
        displaced_local_max = spectrum_local_max[1:]
        index1 = displaced_local_max.argmax() + 1
    else:
        index1 = spectrum_local_max.argmax()
        
    f1 = xf_local_max[index1]
    h1 = spectrum_local_max[index1]

    # Loop checking the multiples of the fundamental
    freqs, heights = [f1], [h1]
    n = 2
    while n:
    
        # Find multiples of the fundamental frequency
        fn = n * f1
        close_index = np.where(np.isclose(xf_local_max,fn, rtol=1e-02))

        if not list(close_index[0]):
            break
    
        freq = xf_local_max[close_index][0]
        height = spectrum_local_max[close_index][0]
    
        # Append the values
        freqs.append(freq)
        heights.append(height)
        
        n += 1
    
        # Stop if no more maximums are found
        if fn > xf_local_max[-1]:
            n = False

    return np.array(freqs), np.array(heights)
    

def inv_irregularity(spectrum):
    "Inverse of the IRR metric, as defined in the reference"
    
    if len(spectrum) == 1:
        return 0.0

    IRR = 0
    for k in range(1,len(spectrum)-1):
        k_contribution = 20 * np.log10(spectrum[k]) 
        k_contribution -= 1/3 * (20 * np.log10(spectrum[k+1])
                               + 20 * np.log10(spectrum[k])
                               + 20 * np.log10(spectrum[k-1]))
        IRR += np.abs(k_contribution)
    
    IRR = np.log10(IRR)
    
    return 1/IRR
    
def measures(xf, spectrum):
    "Compute the spectral center, roughness of the specturm and inverse IRR"
    
    center = spectrum @ xf / np.sum(spectrum)
    roughness = np.std(spectrum)
    iIRR = inv_irregularity(spectrum)
    
    return center, roughness, iIRR




#### ENVELOPE INDICES ####

def envelope_points(tsol, xsol):
    """
    Obtain the peaks (during oscillation) and the points 
    prior and after the oscillations occurs.
    """

    # Obtain peak indices
    peaks = scipy.signal.find_peaks(xsol)[0]
    
    # Obtain the peaks during oscillation
    tp = tsol[peaks]
    xp = xsol[peaks]

    # Get datapoints before and after oscillations
    tbefore = tsol[500:np.where(xsol[100:] > 0.99)[0][0]+100]
    xbefore = xsol[500:np.where(xsol[100:] > 0.99)[0][0]+100]
    
    tafter = tsol[np.where(xsol > 0.99)[0][-1]:]
    xafter = xsol[np.where(xsol > 0.99)[0][-1]:]

    # Add an datapoints before and after oscillations
    tp = np.append(tbefore, tp)
    xp = np.append(xbefore, xp)
    
    tp = np.append(tp, tafter)
    xp = np.append(xp, xafter)

    return tp, xp
