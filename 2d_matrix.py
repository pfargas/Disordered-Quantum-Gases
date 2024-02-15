import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.special import hankel1

from random_lattice import RandomLattice

if not os.path.exists('logs'):
    os.makedirs('logs')

logger_name= "{:%y-%m-%d_%H%M%S}".format(datetime.datetime.now())

logging.basicConfig(filename=f'logs/{logger_name}.log', filemode='w',level=logging.INFO ,format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Starting program')


GAMMA_EULER =  0.57721566490153286060


class Scattering2D:
    
    def __init__(self, lattice_with_dispersors:RandomLattice|None, energy:float, r0:npt.ArrayLike,index_of_resonance:int,effective_scattering:float|None=None,**kwargs):
        self.lattice = lattice_with_dispersors or RandomLattice(kwargs)
        self.k_moment = np.sqrt(2*energy)/self.lattice.spacing
        if effective_scattering is None:
            self.effective_scattering = self.compute_effective_scattering_length_resonance(index_of_localized=index_of_resonance)
        else:
            self.effective_scattering = effective_scattering
        self.M_diag = np.diag(self.diagonal_terms())
        self.Q_matrix = self.off_diagonal_terms()
        self.M_matrix = self.M_diag + self.Q_matrix
        # self.fill_matrix()
        self.free_propagator = np.zeros(self.lattice.n_dispersors, dtype=np.complex128)
        self.r0 = r0
        self.fill_free_propagator()
        self.intensities = self.compute_intensities()
        
        
    # def fill_matrix_v2(self):
    #     try:
    #         for i, (x, y) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
    #             for j, (x_prime, y_prime) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
    #                 if i == j:
    #                     self.M_matrix[i, j] = np.log(self.k_moment*self.effective_scattering*np.exp(GAMMA_EULER))-1j*np.pi/2
    #                 else:
    #                     self.M_matrix[i, j] = -1j*np.pi/4*hankel1(0, self.k_moment*np.sqrt((x-x_prime)**2+(y-y_prime)**2))
    #         logging.info('Matrix filled')
    #     except Exception as e:
    #         logging.error(e)
    #     return 
    
    # def fill_matrix(self):
    #     try:
    #         self.M_matrix = np.diag(self.diagonal_terms()) + self.off_diagonal_terms()
    #         logging.info('Matrix filled')
    #     except Exception as e:
    #         logging.error(e)
    #     return
    
    def fill_free_propagator(self):
        try:
            for i, (x, y) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                self.free_propagator[i] = 1j*np.pi/2*hankel1(0, self.k_moment*np.sqrt((x-self.r0[0])**2+(y-self.r0[1])**2))
            logging.info('Free propagator filled')
        except Exception as e:
            logging.error(e)
        return
    
    def off_diagonal_terms(self):
        Q_matrix = np.zeros((self.lattice.n_dispersors, self.lattice.n_dispersors), dtype=complex)
        try:
            for i, (x, y) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                for j, (x_prime, y_prime) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                    if i != j:
                        Q_matrix[i, j] = -1j*np.pi/4*hankel1(0, self.k_moment*np.sqrt((x-x_prime)**2+(y-y_prime)**2))
            logging.info('Off-diagonal terms computed')
        except Exception as e:
            logging.error(e)
        return Q_matrix
    
    def diagonal_terms(self):
        M_diag = np.zeros(self.lattice.n_dispersors, dtype=complex)
        try:
            for i in range(self.lattice.n_dispersors):
                M_diag[i] = np.log(self.k_moment*self.effective_scattering*np.exp(GAMMA_EULER))-1j*np.pi/2
            logging.info('Diagonal terms computed')
        except Exception as e:
            logging.error(e)
        return M_diag

    def compute_intensities(self):
        try:
            intensities = np.linalg.solve(self.M_matrix, self.free_propagator)
            logging.info('Intensities computed')
        except Exception as e:
            logging.error(e)
        return intensities
    
    def plot_intensities(self):
        try:
            plt.scatter(self.lattice.X_dispersors, self.lattice.Y_dispersors, c=np.abs(self.intensities))
            plt.colorbar()
            plt.show()
            logging.info('Intensities plotted')
        except Exception as e:
            logging.error(e)
        return
    
    def diagonalize_off_diagonal(self):
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.Q_matrix)
            logging.info('Off-diagonal terms diagonalized')
        except Exception as e:
            logging.error(e)
        return eigenvalues, eigenvectors
    
    def find_localized_states(self):
        eigenvalues, eigenvectors = self.diagonalize_off_diagonal()
        index_localized = [
            i
            for i, eigenvalue in enumerate(eigenvalues)
            if np.imag(eigenvalue) < 1e-6
        ]
        return eigenvalues[index_localized], eigenvectors[:, index_localized]
    
    def compute_effective_scattering_length_resonance(self, index_of_localized=0):
        eigvals, _ = self.find_localized_states()
        q_n=eigvals[index_of_localized]
        return 2/self.k_moment*np.exp(-(np.real(q_n)+GAMMA_EULER))