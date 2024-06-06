import numpy as np

def histogram_new(a_eff, z_res, energies, n_a_eff=350, min_ln_a_eff=-2.5, max_ln_a_eff=1.5):
    """Generation of the image
    
    Behaviour:
    1. Even though each index should be the same bin in the energy, check that that is, in fact, true.
    2. Choose the corresponding a_eff bin
    3. Append the imaginary vaule of z_res to the selected bin
    
    Then: each bin should contain an array of imaginary values of z_res. Take the mean in each bin
    
    Inputs:
        a_eff: array of arrays of a_eff values. Each subarray corresponds to a different (initial) energy
        z_res: array of arrays of z_res values. Each subarray corresponds to a different (initial) energy
        energies: array of (initial) energies
    Outputs:
        histogram: array of arrays of the mean imaginary values of z_res in each bin
    """
    delta_a_eff = (max_ln_a_eff - min_ln_a_eff) / n_a_eff
    bins_a_eff = np.linspace(min_ln_a_eff, max_ln_a_eff, n_a_eff)
    delta_energy = energies[1] - energies[0]
    histogram = [[[] for _ in range(n_a_eff)] for _ in range(len(energies))]
    for i in range(len(energies)):
        a_eff_cur = a_eff[i]
        energies_cur = np.real(z_res[i])
        lifetime_cur = np.imag(z_res[i])
        assert len(a_eff_cur) == len(z_res[i])
        log_a_eff = np.real(np.log(a_eff_cur))
        # check that the energy is inside the bin
        energy_index = [None for _ in range(len(a_eff_cur))]
        a_eff_index = [None for _ in range(len(a_eff_cur))]
        for j in range(len(a_eff_cur)):
            energy_index[j] = int((energies_cur[j] - energies[0]) // delta_energy)
            a_eff_index[j] = int((log_a_eff[j] - min_ln_a_eff) // delta_a_eff)
            if a_eff_index[j] > n_a_eff - 1 or energies_cur[j] <0 or a_eff_index[j] < 0:
                continue
            assert energy_index[j] < len(energies)
            assert a_eff_index[j] < n_a_eff
            try:
                histogram[energy_index[j]][a_eff_index[j]].append(lifetime_cur[j])
            except IndexError:
                print("__________________")
                print(energy_index[j])
                print(f"max a_eff index: {n_a_eff - 1}")
                print(a_eff_index[j])
    
    new_histogram = np.zeros((len(histogram[0]), len(histogram[0][0])))
    
    for i in range(len(histogram[0])):
        for j in range(len(histogram[0][0])):
            new_histogram[i][j] = np.mean(histogram[i][j])
    
    return new_histogram