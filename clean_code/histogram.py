import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm as tqdm

class Histogram:
    def __init__(self, results, min_energy, max_energy, n_energies, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff, compute_width=True):
        self.results = results
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.n_energies = n_energies
        self.min_ln_a_eff = min_ln_a_eff
        self.max_ln_a_eff = max_ln_a_eff
        self.n_ln_a_eff = n_ln_a_eff
        self.histogram = np.zeros((n_ln_a_eff, n_energies))
        self.delta_energy = (max_energy-min_energy)/n_energies
        self.delta_a_eff = (max_ln_a_eff-min_ln_a_eff)/n_ln_a_eff
        self.delta_surface = self.delta_energy*self.delta_a_eff
        self.compute_width_histogram() if compute_width else self.compute_histogram()
    def compute_histogram(self):
        for result in tqdm(self.results):
            if result.is_resonance():
                energy_index = result.energy_index(self.min_energy, self.max_energy, self.n_energies)
                a_eff_index = result.a_eff_index(self.min_ln_a_eff, self.max_ln_a_eff, self.n_ln_a_eff)
                if a_eff_index!=-1 and energy_index!=-1:
                    print(f"Energy index: {energy_index}, a_eff index: {a_eff_index}")
                    print(f"Energy: {result.energy}, a_eff: {result.a_eff}")
                    self.histogram[a_eff_index, energy_index]+=1
    def compute_width_histogram(self):
        number_of_resonances = np.zeros((self.n_ln_a_eff, self.n_energies), dtype=int)
        for result in tqdm(self.results):
            energy_index = result.energy_index(self.min_energy, self.max_energy, self.n_energies)
            a_eff_index = result.a_eff_index(self.min_ln_a_eff, self.max_ln_a_eff, self.n_ln_a_eff)
            if a_eff_index!=-1 and energy_index!=-1:
                self.histogram[a_eff_index, energy_index]+=np.log10(result.width)
                number_of_resonances[a_eff_index, energy_index]+=1
        self.histogram = self.histogram/number_of_resonances
        self.histogram = np.nan_to_num(self.histogram)
    def plot(self):
        plt.pcolor(self.histogram)
        plt.colorbar()
        plt.show()
