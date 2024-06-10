import numpy as np
from D_vector import compute_secondary_source_terms

def transmission_coefficient(k:float, theta: float, dispersor_set:np.ndarray, source, a_eff, is_in_radian=True):
    """Computes the transmission coefficient for a given direction theta.
    
    Inputs:
    k: float, the wave number.
    theta: float, the direction at the point.
    
    """
    if not is_in_radian:
        theta = np.deg2rad(theta)
    d_vector = compute_secondary_source_terms(k=k, dispersor_set=dispersor_set, a_eff=a_eff, source=source)
    unitari_direction = np.array([np.sin(theta), np.cos(theta)])
    transmission = 1
    for i, dispersor in enumerate(dispersor_set):
        r0_to_dispersor = dispersor-source
        transmission += d_vector[i]*np.exp(1j*k*unitari_direction@r0_to_dispersor)
    return transmission

def lyapunov_exponent_theta():
    pass