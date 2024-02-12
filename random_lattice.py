import numpy as np
import matplotlib.pyplot as plt
import csv


class RandomLattice:
    def __init__(self, p=0.1, radius=1, spacing=0.04, seed=42):
        self.n = int(radius / spacing)
        self.radius = radius
        self.spacing = spacing
        self.X_lattice, self.Y_lattice = self.generate_circular_lattice()
        self.index = 0
        np.random.seed(seed)  # !!!for reproducibility
        mask = np.random.rand(len(self.X_lattice)) < p
        self.X_dispersors = self.X_lattice[mask]
        self.Y_dispersors = self.Y_lattice[mask]
        self.n = len(self.X_dispersors)

    def generate_circular_lattice(self):

        x = np.linspace(-self.radius, self.radius, self.n)
        y = np.linspace(-self.radius, self.radius, self.n)
        X, Y = np.meshgrid(x, y)

        # mask out the points outside the circle
        R = np.sqrt(X**2 + Y**2)
        mask = R <= self.radius

        # generate a mesh of points in the circle
        X = X[mask]
        Y = Y[mask]
        return X, Y

    def write_to_csv(self, filename):
        with open(filename, mode="w", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Dispersor"])
            for i, _ in enumerate(self.X_lattice):
                if self.X_lattice[i] in self.X_dispersors:
                    mask = self.X_dispersors == self.X_lattice[i]
                    if self.Y_lattice[i] in self.Y_dispersors[mask]:
                        writer.writerow([self.X_lattice[i], self.Y_lattice[i], 1])
                    else:
                        writer.writerow([self.X_lattice[i], self.Y_lattice[i], 0])
                else:
                    writer.writerow([self.X_lattice[i], self.Y_lattice[i], 0])

    # TODO: compute the parameters from the csv file
    @classmethod
    def build_from_csv(cls, filename):
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            X_lattice = []
            Y_lattice = []
            X_dispersors = []
            Y_dispersors = []
            for row in reader:
                X_lattice.append(float(row[0]))
                Y_lattice.append(float(row[1]))
                if int(row[2]) == 1:
                    X_dispersors.append(float(row[0]))
                    Y_dispersors.append(float(row[1]))


if __name__ == "__main__":
    lattice = RandomLattice()
    lattice.write_to_csv("lattice.csv")
    plt.scatter(lattice.X_lattice, lattice.Y_lattice, s=1)
    plt.scatter(lattice.X_dispersors, lattice.Y_dispersors, s=1, c="r")
    plt.show()
    with open("lattice.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            plt.scatter(
                float(row[0]), float(row[1]), s=1, c="r" if int(row[2]) == 1 else "b"
            )
    plt.show()
