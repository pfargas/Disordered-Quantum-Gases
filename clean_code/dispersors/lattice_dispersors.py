import numpy as np

def generate_lattice_dispersors(length, p):
    """Function that generates the dispersors in a lattice of size length x length with probability p.
    Inputs:
    length: int, the size of the lattice.
    p: float, the probability of a dispersor being present in each lattice site.
    """
    dispersors = []
    for i in range(-length, length):
        for j in range(-length, length):
            if np.random.rand() < p:
                dispersors.append([i, j])
    return np.array(dispersors)

def generate_circular_lattice_dispersors(radius,p):
    
    dispersors = generate_lattice_dispersors(radius, p)
    mask = np.linalg.norm(dispersors, axis=1) < radius
    return dispersors[mask]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    radius = 150
    dispersors = generate_circular_lattice_dispersors(radius, 0.1)
    plt.title(f"Number of dispersors: {len(dispersors)}")
    plt.scatter(dispersors[:,0], dispersors[:,1], s = 1)
    # generate a circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    plt.plot(x, y, color="red")
    plt.axis("equal")
    plt.show()
    