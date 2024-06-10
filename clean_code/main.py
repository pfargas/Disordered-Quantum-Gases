from histogram import Histogram
from utils import *

def main():
    length, min_energy, max_energy, n_energies, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff, occupation_probability = read_inputs()
    results = compute_resonances_total(min_energy, max_energy, n_energies, length, occupation_probability)
    histogram = Histogram(results, min_energy, max_energy, n_energies, min_ln_a_eff, max_ln_a_eff, n_ln_a_eff)
    histogram.plot()
    return results
if __name__ == '__main__':
    main()
