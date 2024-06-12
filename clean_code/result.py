import numpy as np

class Result:
    def __init__(self, resonance, a_eff, energy):
        self.resonance = resonance
        self.a_eff = a_eff
        # self.energy = np.real(resonance)
        self.energy = energy
        self.width = np.imag(resonance)*2
    
    def energy_index(self, min_energy, max_energy, n_energies):
        if self.energy>max_energy or self.energy<min_energy:
            return -1
        return int((self.energy-min_energy)/(max_energy-min_energy)*n_energies)
    def a_eff_index(self, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff):
        # Important!!! The a_eff is in log scale
        if np.log(self.a_eff)>max_ln_a_eff or np.log(self.a_eff)<min_ln_a_eff:
            return -1
        return int((np.log(self.a_eff)-min_ln_a_eff)/(max_ln_a_eff-min_ln_a_eff)*n_ln_a_eff)
    def is_resonance(self, tolerance=1e-6):
        return self.width<tolerance