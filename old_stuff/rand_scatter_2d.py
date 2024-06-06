import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.special import hankel1

from random_lattice import RandomLattice

if not os.path.exists("logs"):
    os.makedirs("logs")

logger_name = "{:%y-%m-%d_%H%M%S}".format(datetime.datetime.now())

logging.basicConfig(
    filename=f"logs/{logger_name}.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting program")


GAMMA_EULER = 0.57721566490153286060


class Scattering2D:
    """Class that generates the problem of scattering in 2D
    to find localized states
    """
    def __init__(
        self,
        lattice_with_dispersors: RandomLattice | None,
        energy: float,
        r0: npt.ArrayLike,
        index_of_resonance: int = 0,
        effective_scattering: float | None = None,
        **kwargs,
    ):
        """init method

        Args:
            lattice_with_dispersors (RandomLattice | None): lattice object with dispersors. Defaults to None.
            energy (float): energy of the propagating matter wave
            r0 (npt.ArrayLike): source of the wave
            index_of_resonance (int, optional): index of the searched ressonance state. Defaults to 0.
            effective_scattering (float | None, optional): Effective scattering length of the problem. Defaults to None.
        """
        self.lattice = lattice_with_dispersors or RandomLattice(kwargs)
        self.k_moment = np.sqrt(2 * energy) / self.lattice.spacing
        self.Q_matrix = self.off_diagonal_terms()
        if effective_scattering is None:
            self.effective_scattering = (
                self.compute_effective_scattering_length_resonance(
                    index_of_localized=index_of_resonance
                )
            )
        else:
            self.effective_scattering = effective_scattering
        self.M_diag = np.diag(self.diagonal_terms())
        self.M_matrix = self.M_diag + self.Q_matrix
        # self.fill_matrix()
        self.free_propagator = np.zeros(self.lattice.n_dispersors, dtype=np.complex128)
        self.r0 = r0
        self.fill_free_propagator()
        self.intensities = self.compute_intensities()

    def fill_matrix_v2(self):
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

    def fill_matrix(self):
        try:
            self.M_matrix = np.diag(self.diagonal_terms()) + self.off_diagonal_terms()
            logging.info('Matrix filled')
        except Exception as e:
            logging.error(e)
        return

    def fill_free_propagator(self):
        """Method to generate the 2D free propagator
        """
        try:
            for i, (x, y) in enumerate(
                zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)
            ):
                self.free_propagator[i] = (
                    1j
                    * np.pi
                    / 2
                    * hankel1(
                        0,
                        self.k_moment
                        * np.sqrt((x - self.r0[0]) ** 2 + (y - self.r0[1]) ** 2),
                    )
                )
            logging.info("Free propagator filled")
        except Exception as e:
            logging.error(e)
        return

    def off_diagonal_terms(self):
        Q_matrix = np.zeros(
            (self.lattice.n_dispersors, self.lattice.n_dispersors), dtype=complex
        )
        # try:
        for i, (x, y) in enumerate(
            zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)
        ):
            for j, (x_prime, y_prime) in enumerate(
                zip(self.lattice.X_dispersors, self.lattice.Y_dispersors)
            ):
                if i != j:
                    Q_matrix[i, j] = (
                        -1j
                        * np.pi
                        / 4
                        * hankel1(
                            0,
                            self.k_moment
                            * np.sqrt((x - x_prime) ** 2 + (y - y_prime) ** 2),
                        )
                    )
        logging.info("Off-diagonal terms computed")
        # except Exception as e:
        #     logging.error(e)
        return Q_matrix

    def diagonal_terms(self):
        M_diag = np.zeros(self.lattice.n_dispersors, dtype=complex)
        try:
            for i in range(self.lattice.n_dispersors):
                M_diag[i] = (
                    np.log(
                        self.k_moment * self.effective_scattering * np.exp(GAMMA_EULER)
                    )
                    - 1j * np.pi / 2
                )
            logging.info("Diagonal terms computed")
        except Exception as e:
            logging.error(e)
        return M_diag

    def compute_intensities(self):
        try:
            intensities = np.linalg.solve(self.M_matrix, self.free_propagator)
            logging.info("Intensities computed")
        except Exception as e:
            logging.error(e)
        return intensities
    
    def compute_wavefunction(self, r: npt.ArrayLike):
        """computes the imaginary part of the interacting Green's function

        Args:
            r (npt.ArrayLike): Point at which the wavefunction is computed
        """
        
        # TODO: The free propagators are wrong, they should be computed at the point r
        wavefunction = np.imag(self.free_propagator+np.sum(self.intensities*self.free_propagator))
        return 0

    def plot_intensities(self):
        try:
            plt.scatter(
                self.lattice.X_dispersors,
                self.lattice.Y_dispersors,
                c=np.abs(self.intensities),
            )
            plt.colorbar()
            plt.savefig(f"out/{self.k_moment}.png")
            logging.info("Intensities plotted")
        except Exception as e:
            logging.error(e)
        return

    def diagonalize_off_diagonal(self):
        # try:
        eigval, eigvec = np.linalg.eig(self.Q_matrix)

        logging.info("Off-diagonal terms diagonalized")
        # except Exception as e:
        #     logging.error(e)
        return eigval, eigvec

    def find_localized_states(self):
        eigenvalues, eigenvectors = self.diagonalize_off_diagonal()
        index_localized = [
            int(i)
            for i, eigenvalue in enumerate(eigenvalues)
            if np.imag(eigenvalue) < 1e-6
        ]
        eigenvectors = np.array(eigenvectors)

        loc_eigenvalues = [eigenvalues[i] for i in index_localized]
        loc_eigenvectors = np.array([eigenvectors[:, i] for i in index_localized])
        return loc_eigenvalues, loc_eigenvectors

    def compute_effective_scattering_length_resonance(self, index_of_localized=0):
        eigvals, _ = self.find_localized_states()
        q_n = eigvals[index_of_localized]
        return 2 / self.k_moment * np.exp(-(np.real(q_n) + GAMMA_EULER))


if __name__ == "__main__":
    import multiprocessing as mp
    series_filename = "logs/series_{:%y-%m-%d_%H%M%S}.log".format(datetime.datetime.now())
    energy_vals = [0.0001, 0.001, 0.01, 0.1]
    time = datetime.datetime.now()
    for k in energy_vals:
        lattice = RandomLattice(radius=100, spacing=1)
        scattering = Scattering2D(lattice, energy=k, r0=(0, 0))
        scattering.plot_intensities()
        with open(f"out/{k}.txt", "w", encoding="utf-8") as file:
            file.write(f"Energy {k}, scatter_length={scattering.effective_scattering}\n")
            
    time2 = datetime.datetime.now()
    time_elapsed = time2 - time
    time_elapsed_in_seconds = time_elapsed.total_seconds()
                
    with open(series_filename, "w", encoding="utf-8") as file:
        file.write(f"Series computation took {time_elapsed_in_seconds}")

    lattice = RandomLattice(radius=100, spacing=1)
    # Parallelize the computation of the intensities
    time = datetime.datetime.now()
    with mp.Pool(processes=8) as pool:
        args = [(lattice, k, (0, 0)) for k in energy_vals]
        pool.starmap(Scattering2D, args)
    time2 = datetime.datetime.now()
    
    time_elapsed = time2 - time
    # Write the time elapsed to a file in seconds
    time_elapsed_in_seconds = time_elapsed.total_seconds()
    
    parallel_filename = "logs/parallel_{:%y-%m-%d_%H%M%S}.log".format(time)
    
    with open(parallel_filename, "w", encoding="utf-8") as file:
        file.write(f"Parallel computation took {time_elapsed_in_seconds}")