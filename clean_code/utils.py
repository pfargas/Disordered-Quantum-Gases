import numpy as np
from time import time
from tqdm.autonotebook import tqdm as tqdm
import pickle

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
        with open(filename) as f:
            inputs = json.load(f)
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
    # s_p_rho is the Participation ratio
    a_eff, width, s_p_rho, eigvals, eigvecs = resonances(energy, M_matrix_inf, distances, input)
    s_p = s_p_rho/p
    return a_eff, width, s_p, M_matrix_inf, eigvals, eigvecs

@timer
def compute_resonances_total(input:Input , length, occupation_probability,results=[]):
    settings = input.settings_energy_sweep
    # with open("./out/info_last_exec.txt", "w") as f:
    #     f.write(str(input))
    energies = np.arange(settings["min"], settings["max"], settings["step"])
    for energy in tqdm(energies):
        a_eff, widths, s_p, m_inf, eigvals, eigvecs = compute_resonances_per_energy(energy,input ,length=length, p=occupation_probability)
        for a_eff, width, s_p in zip(a_eff, widths, s_p):
            results.append(Result(imag_resonance=width, 
                              a_eff=a_eff, 
                              energy=energy, 
                              s_p=s_p, 
                              m_inf=m_inf, 
                              spectrum=eigvals, 
                              eigstates=eigvecs)
                           )
    with open('./results.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results