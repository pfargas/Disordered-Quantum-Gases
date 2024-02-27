import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from rand_scatter_2d import Scattering2D
from random_lattice import RandomLattice


def multiprocessing_function(energy):
    lattice = RandomLattice(radius=100, spacing=1, p=0.05)
    scattering = Scattering2D(lattice, energy, r0=(0, 20),effective_scattering=1)
    eigenvalues, _ = scattering.diagonalize_off_diagonal()
    resonances = 0 
    for eigenvalue in eigenvalues:
        if np.abs(np.imag(eigenvalue)) <= 1e-6:
            resonances += 1
    return_dict[energy] = [resonances,len(eigenvalues)]

if __name__ == "__main__":
    # energies = np.linspace(0.01, 0.4, 50)
    # lattice = RandomLattice(radius=100, spacing=1)
    # histogram = {}
    # for energy in energies:
    #     scattering = Scattering2D(lattice, energy, r0=(0, 0),effective_scattering=1)
    #     eigenvalues, _ = scattering.find_localized_states()
    #     resonances = 0 
    #     for eigenvalue in eigenvalues:
    #         if np.imag(eigenvalue) <= 1e-6 and np.real(eigenvalue)<=0.1:
    #             resonances += 1
    #     histogram[energy] = resonances
    
    manager = mp.Manager()
    return_dict = manager.dict()
    energies = np.linspace(0.01, 0.4, 50)
    jobs = []
    for energy in energies:
        p = mp.Process(target=multiprocessing_function, args=(energy,))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    histogram = dict(return_dict)
    
    resonance_number = []
    total_number = []
    
    for i in histogram:
        resonance_number.append(histogram[i][0])
        total_number.append(histogram[i][1])
    
    plt.plot(histogram.keys(), resonance_number, "o")
    plt.plot(histogram.keys(), total_number, "o")
    plt.xlabel("Energy")
    plt.ylabel("Number of localized states")
    plt.show();
