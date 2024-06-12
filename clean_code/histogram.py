import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm as tqdm
from utils import Input

class Histogram:
    def __init__(self, results, settings:Input, compute_width=True):
        self.results = results
        self.min_energy = settings.settings_energy_histogram["min"]
        self.max_energy = settings.settings_energy_histogram["max"]
        self.n_energies = settings.settings_energy_histogram["number"]
        self.min_ln_a_eff = settings.settings_a_eff_histogram["min"]
        self.max_ln_a_eff = settings.settings_a_eff_histogram["max"]
        self.n_ln_a_eff = settings.settings_a_eff_histogram["number"]
        self.histogram = np.zeros((self.n_ln_a_eff, self.n_energies))
        self.delta_energy = settings.settings_energy_histogram["step"]
        self.delta_a_eff = settings.settings_a_eff_histogram["step"]
        self.delta_surface = self.delta_energy*self.delta_a_eff
        print(self.n_energies)
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
                try:
                    self.histogram[a_eff_index, energy_index]+=np.log10(result.width)
                    number_of_resonances[a_eff_index, energy_index]+=1
                except IndexError:
                    print(f"Energy index: {energy_index}, a_eff index: {a_eff_index}")
                    print(f"shapes: Histogram: {self.histogram.shape}")
                    print(f"n_energy: {self.n_energies}, n_a_eff: {self.n_ln_a_eff}")
                    print(f"Energy: {result.energy}, a_eff: {result.a_eff}")
        self.histogram = self.histogram/number_of_resonances
        plt.pcolor(number_of_resonances)
        plt.show()
        self.histogram = np.nan_to_num(self.histogram)
    def plot(self):
        energy = np.linspace(self.min_energy, self.max_energy, self.n_energies)
        ln_a_eff = np.linspace(self.min_ln_a_eff, self.max_ln_a_eff, self.n_ln_a_eff)
        print(f"Shapes: energy: {energy.shape}, ln_a_eff: {ln_a_eff.shape}, histogram: {self.histogram.shape}")
        plt.pcolor(energy, ln_a_eff, self.histogram)
        plt.colorbar()
        plt.show()
