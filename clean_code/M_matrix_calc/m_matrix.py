from numba import njit, prange
import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
from time import time

EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992


def off_diagonal_M(distances, k):
    """Computes the off diagonal elements of the M matrix.
    
    Inputs:
    distances: numpy array of shape (n, n) where n is the number of dispersors. It is a superior triangular matrix with diag and lower part filled with zeros.
    k: float, wavenumber
    """
    # check that distances is 0 in the diagonal and lower part
    mask = np.tri(distances.shape[0], k=0, dtype=bool)
    assert np.all(distances[mask] == 0), "distances should be a superior triangular matrix"
    # First compute only upper triangular part of the M matrix
    argument = k * distances
    total_argument = argument + argument.T
    M_matrix = -1j*np.pi/2*hankel1(0, total_argument)
    # set diag of M_matrix to 0
    np.fill_diagonal(M_matrix, 0.+0.*1j)
    return M_matrix

def off_diag_M_shifted(distances, k, epsilon_k):
    return off_diagonal_M(distances, k+epsilon_k)

def diagonal_M_inf(k, n_dispersors):
    """Compute the diagonal without the effective scattering length (Diagonal elements of M_inf matrix)"""
    diag_value = -1j*np.pi/2+np.log(k/2)+EULER_GAMMA
    return diag_value*np.eye(n_dispersors)

def diagonal_M_inf_shifted(k, n_dispersors, epsilon_k):
    return diagonal_M_inf(k+epsilon_k, n_dispersors)

def a_eff_diagonal(a_eff, n_dispersors):
    """Computes the diagonal part of M depending only on the effective scattering length."""
    return np.log(a_eff)*np.eye(n_dispersors)

def M_inf(k,distances):
    """Computes the total M_inf matrix. (M matrix without the effective scattering length)"""
    return off_diagonal_M(distances, k) + diagonal_M_inf(k, distances.shape[0])

def M_inf_shifted(k, distances, epsilon_k):
    return off_diag_M_shifted(distances, k, epsilon_k) + diagonal_M_inf_shifted(k, distances.shape[0], epsilon_k)

def M_total(k, distances, a_eff):
    """Computes the total M matrix. (M matrix with the effective scattering length)"""
    return M_inf(k, distances) + a_eff_diagonal(a_eff, distances.shape[0])

def derivative_M_inf_k(k, distances):
    """Computes the derivative of the matrix with respect to k"""
    # check that distances is 0 in the diagonal and lower part
    start_time = time()
    mask = np.tri(distances.shape[0], k=0, dtype=bool)
    assert np.all(distances[mask] == 0), "distances should be a superior triangular matrix"
    # print(f"Assertion passed in {time()-start_time:.2f} seconds")
    
    start_time = time()
    argument = k * distances
    argument = argument + argument.T
    hankel_start = time()
    M_matrix = 1j*np.pi/2*hankel1(1, argument)*distances
    # print(f"Hankel computed in {time()-hankel_start:.2f} seconds")
    # change diag to 0
    np.fill_diagonal(M_matrix, 0.+0.*1j)
    assert np.any(np.isnan(M_matrix)) == False, "There are NaNs in the matrix"
    end = time()
    # print(f"Derivative computed in {end-start_time:.2f} seconds")
    return M_matrix + np.eye(distances.shape[0])*1/k

def derivative_M_inf_E(k, distances):
    
    derivative_k_wrt_E = 1/k
    d_M_inf = derivative_M_inf_k(k, distances)
    return d_M_inf*derivative_k_wrt_E
    
if __name__ == "__main__":
    # generate upper triangular matrix
    test_matrix = np.triu(np.random.rand(5,5), k=1)
    mask = np.tri(test_matrix.shape[0], k=0, dtype=bool)
    print(test_matrix[mask])
    assert np.all(test_matrix[mask] == 0), "distances should be a superior triangular matrix"
    print(test_matrix)