import numpy as np
import matplotlib.pyplot as plt
from time import time

from constants import Constants
from histogram import histogram_new
from dispersors import *
from M_matrix_calc.m_matrix import M_total, M_inf
from M_matrix_calc.utils import distances_between_dispersors
from M_matrix_calc.M_matrix_visualization import plot_real_imag_M_matrix
from M_matrix_calc.resonances import resonances, resonance


# define the constants
GAMMA_EULER = Constants.GAMMA_EULER


def histogram(bins, data):
    """Equispaced serialized histogram
    Input:
        bins: vector containing the maximum value of the bin
        data: data to make the histogram from
    Output:
        histogram: vector with the value of the density (not normalized)
    """
    
    h = bins[1]-bins[0]
    histogram = np.zeros(bins.shape[0])
    for value in data:
        for i,bin in enumerate(bins):
            if value<bin and value>(bin-h):
                # data.pop(value)
                histogram[i]+=1
    return histogram

def graphic_sol_ressonance():
    index = 0
    energy = np.linspace(0.01)
    # Generate a set of dispersors
    length = 100
    fig, ax = plt.subplots(1,2)
    energies = np.linspace(0.005, 0.02, 300)
    start = time()
    k = np.sqrt(2*energy)
    dispersor_set = generate_lattice_dispersors(length=length, p=0.1)
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    M_matrix_inf = M_inf(k=k, distances=distances)
    eigvals, eigvecs = np.linalg.eig(M_matrix_inf)
    eigval = eigvals[index]
    # def find_pole(initial_guess=4+5j):
    # # Define the function whose root we want to find
    # def target_function(z):
    #     return np.eig.eigvals(M_matrix_inf)
    
    # # Use a root-finding algorithm to find the roots
    # root = sp.optimize.root_scalar(target_function, x0=initial_guess, method='newton')
    # return root.root if root.converged else None


    end = time()
    print(f"Time elapsed: {end-start:.2f}")
               
def test_histogram():
    total_a_eff=[]
    total_z_res=[]
    energies = np.linspace(0.01, 0.6, 300)
    # Generate a set of dispersors
    length = 100
    start = time()
    for energy in energies:
        k = np.sqrt(2*energy)
        dispersor_set = generate_lattice_dispersors(length=length, p=0.1)
        distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
        distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
        M_matrix_inf = M_inf(k=k, distances=distances)
        a_eff, z_res = resonances(energy, M_matrix_inf)
        # append values to total arrays but they have to be unpacked
        total_a_eff.append(a_eff)
        total_z_res.append(z_res)
    end = time()
    print(f"Time elapsed: {end-start:.2f}")
    histogram_total = histogram_new(total_a_eff, total_z_res, energies, n_a_eff=100)
    plt.pcolor(histogram_total)
    plt.show()

def compute_resonances_per_energy(energy, length = 70, p=0.1):
    k = np.sqrt(2*energy)
    dispersor_set = generate_lattice_dispersors(length=length, p=p)
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    M_matrix_inf = M_inf(k=k, distances=distances)
    a_eff, z_res = resonances(energy, M_matrix_inf, distances)
    return a_eff, z_res

def main():
    
    import json
    with open("clean_code/inputs.json") as f:
        inputs = json.load(f)
    length = inputs["length"]
    min_energy = inputs["energy"]["min"]
    max_energy = inputs["energy"]["max"]
    n_energies = inputs["energy"]["number"]
    min_ln_a_eff = inputs["ln_a_eff"]["min"]
    max_ln_a_eff = inputs["ln_a_eff"]["max"]
    n_ln_a_eff = inputs["ln_a_eff"]["number"] # number of bins in vertical axis
    occupation_probability = inputs["p"]
    
    total_a_eff=[]
    total_z_res=[]
    energies = np.linspace(min_energy, max_energy, n_energies)
    start = time()
    for energy in energies:
        a_eff, z_res = compute_resonances_per_energy(energy, length=length, p=occupation_probability)
        total_a_eff.append(a_eff)
        total_z_res.append(z_res)
    end = time()
    print(f"Time elapsed: {end-start:.2f}")
    # bins = np.linspace(-1.3,1.5, 300)
    bins = np.linspace(min_ln_a_eff,max_ln_a_eff, n_ln_a_eff)
    histogram_total=[]
    for i in range(len(energies)):
        a_eff = total_a_eff[i]
        z_res = total_z_res[i]
        mask = [np.imag(value)<1e-6 for value in z_res] # Careful the factor of 2 in the definition of ressonance!! Not yet implemented
        a_eff = np.log(np.array(a_eff[mask]))
        histogram_in_energy = histogram(bins, a_eff)
        histogram_total.append(histogram_in_energy)
        # plt.title(f"Histogram of effective scattering length for energy {energies[i]}")
        # plt.plot(bins, histogram_in_energy)
        # plt.savefig(f"energy{i}")
        # plt.clf()
    
    histogram_total = np.array(histogram_total).T
    delta_energy = energies[1]-energies[0]
    delta_a_eff = bins[1]-bins[0]
    delta_surface = delta_energy*delta_a_eff
    # color = np.log10(histogram_total/(delta_surface*0.1*length**2))
    plt.pcolor(energies, bins, histogram_total)
    plt.show()
    
if __name__ == '__main__':
    main()
    # test_histogram()