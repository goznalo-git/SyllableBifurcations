import numpy as np

def airflow(t, lambda1, lambda2, T1=1, T2=2):
    "Amount of air inserted to the system"
    
    opening = 1/(1 + np.exp(-lambda1 * (t - T1)))
    closing = 1/(1 + np.exp(lambda2 * (t - T2)))
    
    return opening * closing


def TakensBogdanov(t, y, mu1, mu2):
    """
    Takens-Bogdanov bifurcation, with x=y[0], y=y[1]
    """
    return y[1], -mu1 -mu2 * y[0] - y[0]**3 - y[0]**2 * y[1] + y[0]**2 - y[0]*y[1]


def VanderPolD_double(t, y, mu, k, delta):
    """
    Modified Van der Pol oscillator, with x=y[0], y=y[1]
    """
    
    return mu * (y[1] - (1/3) * y[0]**3 + y[0]), (- k * y[0] - delta)/mu


def VanderPolD_single(t, y, mu, k, delta):
    """
    Modified Van der Pol oscillator, with x=y[0], y=y[1]
    """
    
    return mu * (y[1] - (1/3) * y[0]**3 + y[0]), - k * y[0] - delta


def Watery(t, y, eps, a, prop=1000):
    """
    Watery Van der Pol oscillator, with x=y[0], y=y[1]
    """
    
    return prop * (y[1] - (y[0]**2 - y[0]**3 + y[0]**5)), prop * (eps * (a - y[0]))


def Watery_timescale(t, y, eps, a, prop=1000):
    """
    Watery Van der Pol oscillator, with x=y[0], Y=y[1]
    """
    
    return prop**2 * y[1] - prop * (y[0]**2 - y[0]**3 + y[0]**5), eps * (a - y[0])


def Hopf_polar(t, y, w, mu, prop=10): 
    """
    Hopf bifurcation, with r=y[0], theta=y[1]
    """
    return prop * (mu - y[0]**2) * y[0], prop * w


def Hopf_cartesian(t, y, w, mu, prop=10): 
    """
    Hopf bifurcation, with x=y[0], y=y[1]
    """
    return prop * ((mu - y[0]**2 - y[1]**2) * y[0] - w * y[1]), prop * ((mu - y[0]**2 - y[1]**2) * y[1] + w * y[0])


def SNILC(t, y, w, mu, prop=10): 
    """
    SNILC bifurcation, with r=y[0], theta=y[1]
    """
    return prop * y[0] * (mu - y[0]**2), prop * (w - np.cos(y[1]))