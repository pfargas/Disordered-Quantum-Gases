import numpy as np
from scipy.special import hankel1
from m_matrix import M_total
from utils import distances_between_dispersors

EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992

def solve_linear_system(M_matrix, b_vector):
    """Solve a linear system of equations.
    
    Inputs:
    M_matrix: numpy array of shape (n, n) where n is the number of dispersors.
    b_vector: numpy array of shape (n,) where n is the number of dispersors.
    """
    return np.linalg.solve(M_matrix, b_vector)

def compute_b_vector(k, dispersor_set, source):
    """Compute the b vector of the linear system, where the linear system is described as
    M_matrix * strengths = b_vector.
    Inputs:
        k: float, wavenumber
        dispersor_set: numpy array of shape (n, 2) where n is the number of dispersors. Each row contains the x and y coordinates of a dispersor.
        source: numpy array of shape (2,) containing the x and y coordinates of the source.
    Outputs:
        b_vector: numpy array of shape (n,) where n is the number of dispersors.
    """
    argument = k*np.linalg.norm(dispersor_set - source, axis=1)
    return -1j*np.pi/2*hankel1(0, argument)

def compute_secondary_source_terms(k, dispersor_set,  a_eff, source):
    """Solves the linear system of equations that describes the problem
    
    Inputs:
        k: float, wavenumber
        dispersor_set: numpy array of shape (n, 2) where n is the number of dispersors. Each row contains the x and y coordinates of a dispersor.
        a_eff: float, effective scattering length.
        source: numpy array of shape (2,) containing the x and y coordinates of the source.
    Outputs:
        b_vector: numpy array of shape (n,) where n is the number of dispersors.
    """
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances, dispersor_set=dispersor_set)
    M_matrix = M_total(k=k, distances=distances, a_eff=a_eff)
    b_vector = compute_b_vector(k=k, dispersor_set=dispersor_set, source=source)
    return solve_linear_system(M_matrix=M_matrix, b_vector=b_vector)