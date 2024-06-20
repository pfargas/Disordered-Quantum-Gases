import numpy as np
from .m_matrix import derivative_M_inf_E
from tqdm.autonotebook import tqdm as tqdm
from time import time

def newton_step(eigenvalue,eigenvector, derived_matrix, energy, index_eigenval,debug=True):
    
    # apply hellmann-feynman theorem
    derivative_eigval = eigenvector.T.conj() @ derived_matrix @ eigenvector
    
    # compute the actual step
    
    step = -1j*(np.imag(eigenvalue)/derivative_eigval)
    if debug:
        with open("./debug.txt", "a") as f:
            # , eigenvalue: {eigenvalue}, derivative_eigval: {derivative_eigval}, step: {step}, ressonance: {2*step},
            f.write(f"Energy:{energy}, eigval: {index_eigenval}, z_res: {energy+step}\n")
    return step

def resonance(energy, index_eigenval, eigvals, eigvecs, derived_matrix):
    """Compute a single pole of the green's function, meaning given an energy and a dispersor lattice, select one eigenvalue and compute the effective scattering length and the energy of the resonance, such that the resonance is a pole of the green's function, i.e. z_{res}=0.
    
    Inputs:
    energy: float, energy of the resonance
    M_inf: numpy array of shape (n, n) where n is the number of dispersors. It is the M_inf matrix of the system.
    index_eigenval: int, index of the eigenvalue to compute the resonance for.
    Outputs:
    a_eff: float, effective scattering length (such that the real part of M is 0 at the ressonance)
    z_res: float, energy of the resonance (complex!)
    """

    eigval = eigvals[index_eigenval]
    eigvec = eigvecs[:,index_eigenval]

    a_eff = float(np.exp(-np.real(eigval)))
    
    s_p_rho = 0
    
    for d_i in eigvec:
        s_p_rho += (d_i)**4
    s_p_rho = 1/s_p_rho

    step = newton_step(eigval, eigvec, derived_matrix, energy, index_eigenval)
    width_2= np.float128(np.imag(step))
    return a_eff, width_2, s_p_rho


def resonances(energy, M_inf,distances, input):
    """Given an energy and a M_inf matrix, compute all resonances of the system"""    
    width = np.zeros(M_inf.shape[0], dtype=np.float128)
    a_eff = np.zeros(M_inf.shape[0], dtype=np.float16)
    s_p_rho = np.zeros(M_inf.shape[0], dtype=np.float16)

    start = time()
    eigvals, eigvecs = np.linalg.eig(M_inf) 
    end = time()

    # sanity check
    i=np.random.randint(M_inf.shape[0])
    assert np.linalg.norm(M_inf@eigvecs[:,i] - eigvals[i]*eigvecs[:,i]) < 1e-5, "Eigenvalues and eigenvectors are not consistent"
    print(f"Energy:{energy}, diagonalization time: {end-start:.2f} s")
    k = np.sqrt(2*energy)
    derived_matrix = derivative_M_inf_E(k, distances)
    for i in tqdm(range(M_inf.shape[0]), desc=f"Diagonalization time: {end-start:.2f}\n Computing resonances", leave=True):
        a_eff[i], width[i], s_p_rho[i] = resonance(energy, i, eigvals, eigvecs, derived_matrix)
    return a_eff, width, s_p_rho, eigvals, eigvecs