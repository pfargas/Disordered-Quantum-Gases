import numpy as np
from .m_matrix import derivative_M_inf_E
from tqdm.autonotebook import tqdm as tqdm
from time import time

def newton_step(eigenvalue,eigenvector, k, distances):
    
    derived_matrix = derivative_M_inf_E(k, distances)
    
    # apply hellmann-feynman theorem
    derivative_eigval = eigenvector.T.conj() @ derived_matrix @ eigenvector
    
    # compute the actual step
    
    step = -1j*(np.imag(eigenvalue)/derivative_eigval)
    
    return step

def resonance(energy, index_eigenval, eigvals, eigvecs, distances, newton=True):
    """Compute a single pole of the green's function, meaning given an energy and a dispersor lattice, select one eigenvalue and compute the effective scattering length and the energy of the resonance, such that the resonance is a pole of the green's function, i.e. z_{res}=0.
    
    Inputs:
    energy: float, energy of the resonance
    M_inf: numpy array of shape (n, n) where n is the number of dispersors. It is the M_inf matrix of the system.
    index_eigenval: int, index of the eigenvalue to compute the resonance for.
    Outputs:
    a_eff: float, effective scattering length (such that the real part of M is 0 at the ressonance)
    z_res: float, energy of the resonance (complex!)
    """

    k = np.sqrt(2*energy)
    eigval = eigvals[index_eigenval]
    eigvec = eigvecs[:,index_eigenval]

    a_eff = float(np.exp(-np.real(eigval)))

    if newton:
        step = newton_step(eigval, eigvec, k, distances)
        return a_eff, step
    else:
        return a_eff, energy + 1j*np.imag(eigval)

def resonances(energy, M_inf,distances):
    """Given an energy and a M_inf matrix, compute all resonances of the system"""    
    z_res = np.zeros(M_inf.shape[0], dtype=np.complex128)
    a_eff = np.zeros(M_inf.shape[0], dtype=np.complex128)

    start = time()
    eigvals, eigvecs = np.linalg.eig(M_inf) 
    end = time()
    
    # sanity check
    i=np.random.randint(M_inf.shape[0])
    assert np.linalg.norm(M_inf@eigvecs[:,i] - eigvals[i]*eigvecs[:,i]) < 1e-5, "Eigenvalues and eigenvectors are not consistent"
    print(f"Energy:{energy}, diagonalization time: {end-start:.2f} s")
    # for i in tqdm(range(M_inf.shape[0]), desc=f"Diagonalization time: {end-start:.2f}\n Computing resonances", leave=True):
    for i in range(M_inf.shape[0]):
        a_eff[i], z_res[i] = resonance(energy, i, eigvals, eigvecs, distances)
    return a_eff, z_res
