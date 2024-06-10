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

def generate_circular_lattice_dispersors(radius,p):
    
    dispersors = generate_lattice_dispersors(radius//2, p)
    mask = np.linalg.norm(dispersors, axis=1) < radius//2
    return dispersors[mask]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dispersors = generate_circular_lattice_dispersors(200, 0.5)
    plt.scatter(dispersors[:,0], dispersors[:,1], s = 1)
    plt.show()
    