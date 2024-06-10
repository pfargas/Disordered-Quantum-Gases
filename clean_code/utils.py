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

def read_inputs():
    import json
    with open("./inputs.json") as f:
        inputs = json.load(f)
    length = inputs["length"]
    min_energy = inputs["energy"]["min"]
    max_energy = inputs["energy"]["max"]
    n_energies = inputs["energy"]["number"]
    min_ln_a_eff = inputs["ln_a_eff"]["min"]
    max_ln_a_eff = inputs["ln_a_eff"]["max"]
    n_ln_a_eff = inputs["ln_a_eff"]["number"] # number of bins in vertical axis
    occupation_probability = inputs["p"]
    return length, min_energy, max_energy, n_energies, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff, occupation_probability


def compute_resonances_per_energy(energy, length = 70, p=0.1):
    k = np.sqrt(2*energy)
    dispersor_set = generate_lattice_dispersors(length=length, p=p)
    distances = np.zeros((dispersor_set.shape[0], dispersor_set.shape[0]))
    distances = distances_between_dispersors(distances_matrix=distances, dispersor_set=dispersor_set)
    M_matrix_inf = M_inf(k=k, distances=distances)
    a_eff, z_res = resonances(energy, M_matrix_inf, distances)
    return a_eff, z_res

@timer
def compute_resonances_total(min_energy, max_energy, n_energies, length, occupation_probability, results=[]):
    energies = np.linspace(min_energy, max_energy, n_energies)
    for energy in tqdm(energies):
        a_eff, z_res = compute_resonances_per_energy(energy, length=length, p=occupation_probability)
        for z_res, a_eff in zip(z_res, a_eff):
            results.append(Result(z_res, a_eff))
    return results