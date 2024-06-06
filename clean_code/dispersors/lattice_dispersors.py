import numpy as np

def generate_lattice_dispersors(length, p):
    """Function that generates the dispersors in a lattice of size length x length with probability p.
    Inputs:
    length: int, the size of the lattice.
    p: float, the probability of a dispersor being present in each lattice site.
    """
    dispersors = []
    for i in range(-length//2, length//2):
        for j in range(-length//2, length//2):
            if np.random.rand() < p:
                dispersors.append([i, j])
    return np.array(dispersors)