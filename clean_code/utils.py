import numpy as np
from time import time
from tqdm.autonotebook import tqdm as tqdm

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
        # length = inputs["length"]
        # min_energy = inputs["energy_bin"]["min"]
        # max_energy = inputs["energy_bin"]["max"]
        # n_energies = inputs["energy_bin"]["number"]
        # min_ln_a_eff = inputs["ln_a_eff"]["min"]
        # max_ln_a_eff = inputs["ln_a_eff"]["max"]
        # n_ln_a_eff = inputs["ln_a_eff"]["number"] # number of bins in vertical axis
        # n_energies_sweep = inputs["energy_sweep"]["number"]
        # occupation_probability = inputs["p"]
        # print(f"Delta energy: {(max_energy-min_energy)/n_energies}")
        # print(f"Delta ln(a_eff): {(max_ln_a_eff-min_ln_a_eff)/n_ln_a_eff}")
        # return length, min_energy, max_energy, n_energies, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff, occupation_probability, n_energies_sweep

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

def compute_resonances_per_energy(energy, length = 70, p=0.1):
    k = np.sqrt(2*energy)
    # dispersor_set = generate_lattice_dispersors(length=length, p=p)
    dispersor_set = generate_circular_lattice_dispersors(radius=length, p=p)
    print(f"Number of dispersors: {dispersor_set.shape[0]}")
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    M_matrix_inf = M_inf(k=k, distances=distances)
    a_eff, z_res = resonances(energy, M_matrix_inf, distances)
    return a_eff, z_res

@timer
def compute_resonances_total(input_sweep:Input , length, occupation_probability,results=[]):
    settings = input_sweep.settings_energy_sweep
    energies = np.arange(settings["min"], settings["max"], settings["step"])
    print(f"Energy step: {settings['step']}, Actual energy step: {energies[1]-energies[0]}")
    for energy in tqdm(energies):
        a_eff, z_res = compute_resonances_per_energy(energy, length=length, p=occupation_probability)
        for z_res, a_eff in zip(z_res, a_eff):
            results.append(Result(z_res, a_eff, energy=energy))
    return results