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
    
    def __init__(self, lattice_with_dispersors:RandomLattice|None, energy:float, effective_scattering:float, r0:npt.ArrayLike,**kwargs):
        self.lattice = lattice_with_dispersors or RandomLattice(kwargs)
        self.k_moment = np.sqrt(2*energy)/self.lattice.spacing
        self.effective_scattering = effective_scattering
        self.M_matrix = np.zeros((self.lattice.n_dispersors, self.lattice.n_dispersors), dtype=complex)
        self.fill_matrix()
        self.free_propagator = np.zeros(self.lattice.n_dispersors, dtype=np.complex128)
        self.r0 = r0
        self.fill_free_propagator()
        self.intensities = self.compute_intensities()
        
        
    def fill_matrix(self):
        try:
            for i, (x, y) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                for j, (x_prime, y_prime) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                    if i == j:
                        self.M_matrix[i, j] = np.log(self.k_moment*self.effective_scattering*np.exp(GAMMA_EULER))-1j*np.pi/2
                    else:
                        self.M_matrix[i, j] = -1j*np.pi/4*hankel1(0, self.k_moment*np.sqrt((x-x_prime)**2+(y-y_prime)**2))
            logging.info('Matrix filled')
        except Exception as e:
            logging.error(e)
        return 
    
    def fill_free_propagator(self):
        try:
            for i, (x, y) in enumerate(zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)):
                self.free_propagator[i] = 1j*np.pi/2*hankel1(0, self.k_moment*np.sqrt((x-self.r0[0])**2+(y-self.r0[1])**2))
            logging.info('Free propagator filled')
        except Exception as e:
            logging.error(e)
        return

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