import numpy as np
from time import time
from tqdm.autonotebook import tqdm as tqdm
import pickle
import os
from scipy.special import hankel1

from result import Result
from dispersors import *
from M_matrix_calc.m_matrix import M_inf
from M_matrix_calc.utils import distances_between_dispersors
from M_matrix_calc.resonances import resonances


def timer(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Time elapsed: {end-start:.2f}")
        return result
    return wrapper

class Input:
    def __init__(self, filename="./inputs.json"):
        self.inputs = self.read_inputs(filename)
        
        self.settings_energy_histogram = self.inputs["energy_bin"]
        if self.settings_energy_histogram["step"] != 0.0:
            self.settings_energy_histogram["number"] = int((self.settings_energy_histogram["max"]-self.settings_energy_histogram["min"])/self.settings_energy_histogram["step"])
        self.length = self.inputs["length"]
        self.settings_a_eff_histogram = self.inputs["ln_a_eff"]
        if self.settings_a_eff_histogram["step"] != 0.0:
            self.settings_a_eff_histogram["number"] = int((self.settings_a_eff_histogram["max"]-self.settings_a_eff_histogram["min"])/self.settings_a_eff_histogram["step"])
        self.occupation_probability = self.inputs["p"]
        self.settings_energy_sweep = self.inputs["energy_sweep"]
        if self.settings_energy_sweep["step"] != 0.0:
            self.settings_energy_sweep["number"] = int((self.settings_energy_sweep["max"]-self.settings_energy_sweep["min"])/self.settings_energy_sweep["step"])

    def read_inputs(self, filename):
        import json
        try:
            with open(filename) as f:
                inputs = json.load(f)
        except FileNotFoundError:
            print("File not found")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in the current directory: {os.listdir()}")
        return inputs

    def __str__(self):
        
        string = f"""
        Length of the system:{self.length}\n
        ______________________\n
        Histogram settings:\n
        *PRIORITY IS GIVEN TO THE STEPS NOT TO THE NUMBER OF BINS*\n
        \t Energy settings:\n
        \t\t Min energy: {self.settings_energy_histogram["min"]}\n
        \t\t Max energy: {self.settings_energy_histogram["max"]}\n
        \t\t Number of bins: {self.settings_energy_histogram["number"]}\n
        \t\t Step size: {self.settings_energy_histogram["step"]}\n
        \t ln(a_eff) settings:\n
        \t\t Min ln(a_eff): {self.settings_a_eff_histogram["min"]}\n
        \t\t Max ln(a_eff): {self.settings_a_eff_histogram["max"]}\n
        \t\t Number of bins: {self.settings_a_eff_histogram["number"]}\n
        \t\t Step size: {self.settings_a_eff_histogram["step"]}\n
        Energy calculation settings:\n
        \t Number of energies to sweep: {self.settings_energy_sweep["number"]}\n
        \t Min energy: {self.settings_energy_sweep["min"]}\n
        \t Max energy: {self.settings_energy_sweep["max"]}\n
        \t Step size: {self.settings_energy_sweep["step"]}\n
        Occupation probability: {self.occupation_probability}
        """
        
        return string

def compute_resonances_per_energy(energy,input:Input ,length = 70, p=0.1):
    k = np.sqrt(2*energy)
    # dispersor_set = generate_lattice_dispersors(length=length, p=p)
    dispersor_set = generate_circular_lattice_dispersors(radius=length, p=p)
    print(f"Number of dispersors: {dispersor_set.shape[0]}")
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    M_matrix_inf = M_inf(k=k, distances=distances)
    np.savez_compressed(f"./M_inf_{energy:.4f}.npz", M_matrix_inf)
    # s_p_rho is the Participation ratio
    a_eff, width, s_p_rho, eigvals, eigvecs = resonances(energy, M_matrix_inf, distances, input)
    s_p = s_p_rho/p
    return a_eff, width, s_p, eigvals, eigvecs

@timer
def compute_resonances_total(input:Input , length, occupation_probability,results=[]):
    settings = input.settings_energy_sweep
    # with open("./out/info_last_exec.txt", "w") as f:
    #     f.write(str(input))
    energies = np.arange(settings["min"], settings["max"], settings["step"])
    for energy in tqdm(energies):
        a_eff, widths, s_p,_,_ = compute_resonances_per_energy(energy,input ,length=length, p=occupation_probability)
        for a_eff, width, s_p in zip(a_eff, widths, s_p):
            results.append(Result(imag_resonance=width, 
                              a_eff=a_eff, 
                              energy=energy, 
                              s_p=s_p)
                           )
    with open('./results.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results

def wavefunctions(positions, energies, input:Input, ln_a_eff=0 ,radius=150, p=0.1):
    """
    Inputs:
    positions: list of floats, positions in which the wavefunction will be computed
    energies: list of floats, energies in which the wavefunction will be computed. The output will be a wavefunction for each energy
    ln_a_eff: float, natural logarithm of the effective scattering length that one want to explore
    input: Input object, contains the settings for the simulation
    radius: float, radius of the circular lattice
    p: float, occupation probability of the dispersors
    
    Output:
    wavefunctions: array of shape (len(positions), len(energies)), contains the wavefunction for each energy in columns (first column is the wavefunction for the first energy in energies, and so on)
    """
    wavefunctions = np.zeros((len(positions), len(energies)), dtype=np.complex128)
    dispersor_set = generate_circular_lattice_dispersors(radius=radius, p=p)
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    global_result = []
    a_eff_slice=0
    index_a_eff = Result(imag_resonance=1e-6, a_eff=np.exp(ln_a_eff), energy=0.1, s_p=0.0).a_eff_index(-2.5, 2.5, 714)
    for energy_index, energy in enumerate(energies):
        k = np.sqrt(2*energy)
        # Compute matrix M_inf
        M_matrix_inf = M_inf(k=k, distances=distances)
        # Diagonalize matrix and get widths
        a_eff, widths, _, eigvals, eigvecs = resonances(energy, M_matrix_inf, distances, input)
        # Loop over the results
        for i,(a_eff, width) in enumerate(zip(a_eff, widths)):
            mock_result = Result(imag_resonance=width, a_eff=a_eff, energy=energy, s_p = 0.0)
            # if the result is inside the slice of a_eff that we want and it is a resonance
            if mock_result.a_eff_index(-2.5,2.5,714)==index_a_eff:
                # global_result is a list of tuples (Result, index), in which there are only values with the given scattering length and the index
                global_result.append([Result(imag_resonance=width, a_eff = a_eff, energy=energy, s_p=0.0), i])
        # CHOOSE THE MINIMUM OR MAXIMUM WIDTH
        minimum = True
        if minimum:
            min_width = 1e-6
            current_idx=0
            for result, idx in global_result:
                if result.width<min_width and result.width>0:
                    min_width = result.width
                    current_idx = idx
            print(min_width)
            final_width = min_width
        else:
            max_width = 0.0
            current_idx=0
            for result, idx in global_result:
                print(f"width:{result.width:.2e}")
                if -result.width>max_width:
                    max_width = result.width
                    current_idx = idx
                    print(f"max_width:{max_width:.2e}")
            print(max_width)
            final_width = max_width
        wavefunctions[:,energy_index] = wavefunction(positions, k, eigvecs[:,current_idx], dispersor_set)
    print(len(global_result))
    return wavefunctions, final_width

def wavefunction(r_vec, k, eigenvector, dispersors):
    wavefunction = np.zeros(r_vec.shape[0], dtype=np.complex128)
    for i,r in tqdm(enumerate(r_vec)):
        distances = np.linalg.norm(dispersors-r, axis=1)
        free_propagator = hankel1(0, k*distances)
        wavefunction[i] = -1j/2*np.sum(free_propagator*eigenvector)
    # normalization
    wavefunction = wavefunction/np.linalg.norm(wavefunction)
    return wavefunction

def generate_points_lattice(length):
    points = []
    for i in range(-length, length):
        for j in range(-length, length):
            points.append([i-0.5,j-0.5])
    return np.array(points)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #test wavefunction
    with open("./debug.txt", "w") as f:
        f.write("")
    with open("./widths.txt", "w") as f:
        f.write("")
    inputs = Input("./clean_code/inputs.json")
    sys_size=80
    r_vector = generate_points_lattice(sys_size+10)
    energies = np.array([0.005])#, 0.06, 0.1])
    wf,width = (wavefunctions(r_vector,energies, input=inputs, radius=sys_size, p=0.1))
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    scatter_1=ax[0].scatter(r_vector[:,0], r_vector[:,1], c=(np.abs(wf[:,0])**2), s=2)
    scatter_2=ax[1].scatter(r_vector[:,0], r_vector[:,1], c=np.log10(np.abs(wf[:,0])**2), s=1.5)
    theta = np.linspace(0, 2*np.pi, 100)
    ax[0].plot(sys_size*np.cos(theta), sys_size*np.sin(theta), c="black")    
    ax[1].plot(sys_size*np.cos(theta), sys_size*np.sin(theta), c="black")
    ax[0].axis("equal")
    ax[1].axis("equal")
    fig.colorbar(scatter_1, ax=ax[0], label="$|\psi (r)|^2$")
    fig.colorbar(scatter_2, ax=ax[1], label="$\log_{10}(|\psi (r)|^2)$")
    ax[0].set_xlabel("$x/d$")
    ax[0].set_ylabel("$y/d$")
    ax[1].set_xlabel("$x/d$")
    ax[1].set_ylabel("$y/d$")
    plt.suptitle(f"$\Gamma=$ {width:.2e}")
    plt.show()