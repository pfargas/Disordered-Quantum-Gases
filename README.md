# Disordered-Quantum-Gases
Study of Anderson localisation in a 2 dimensional disordered quantum gas.

The key object is the $M$ matrix, which can be decomposed as follows:

$$M = \ln a_{eff} \mathbb{I}+M^{\infty}$$

Where $M^{\infty}$ doesn't depend on $a_{eff}$. The key point is that when the energies are extended to the imaginary plane, one can search for complex energies $z_{res}$ that correspond to a pole of the propagator. As the propagator depends on $M^{-1}$, if we express the $M$ matrix as the change of basis from the diagonal to the actual, we see that the poles correspond to some eigenvalue being 0: $m_i=0\rightarrow$ Pole.

Then the problem is to find an energy $z_{res}$ such that $m_i(z_{res})=0$. To do this, we use Newton's method:
1. Choose an energy $E$. This is our initial guess.
2. Take $a_{eff}$ such that: $\ln{a_eff}=-Re{m_i(E)}$
3. The energy that we find is the first Newton step:
   $$z_{res}=E-\text{i}\frac{Im(m^{\infty}_i(E))}{\frac{dm_i^{\infty}}{dE}}$$

Writting explicitly $M^{infty}$ we have:

$$M^{\infty}_{ij}=\begin{cases}
    -\frac{\text{i}\pi}{2}H_0^{(1)}(k|r_i-r_j|)& i\neq j\\
    -\text{i}\frac{\pi}{2}+\ln(\frac{ke^{\gamma}}{2})& i=j
\end{cases}$$

The Hellmann-Feynman theorem states that to compute the derivative of some eigenvalue, one can use (supposing symmetric matrix, i.e. left and write eigvectors fulfill $u^*_i=v_i$):

$$\frac{dm_i^{\infty}(E)}{dE}=v_i^{(T)}\frac{dM^{\infty}(E)}{dE}v_i$$

Where $v_i$ is the eigenvector associated with the eigenvalue.

Then we can compute the derivative of the matrix (component):

$$\frac{dM^\infty(E)}{dE}=\frac{dM^{\infty}(E)}{dk}\frac{dk}{dE}=\frac{dM^{\infty}(E)}{dk}\frac{1}{\sqrt{2E}}=\frac{dM^{\infty}(E)}{dk}\frac{1}{k}$$

Then, we can compute the derivative of each term of $M^\infty$ with respect to $k$:

$$\frac{dM^\infty}{dk}=\begin{cases}
    \frac{\text{i}\pi}{2}H_1^{(1)}(k|r_i-r_j|)\cdot |r_i-r_j| &i\neq j\\
    \frac{1}{k}& i=j
\end{cases}$$

Where it has been used that:

$$\frac{dH_0^{(1)}(k r_{ij})}{dk}=\frac{dH_0^{(1)}(k r_{ij})}{d(kr_ij)}\frac{d (kr_{ij})}{dk}=-H_1^{(1)}(kr_{ij})r_{ij}$$

## Code

The code can be found in the folder `clean_code`. The structure is the following:

- In `main.py` there is the main function. 
- Inside the folder `dispersors/` there is the dispersor generator
- In the `M_matrix_calc/` directory, one can find the actual numerical code.

The objective is to find ressonances. Procedure in the code:

- When `python main.py` is executed, the `main` function (only that) is executed. The main function what does is: 
  - It generates a list of energies to explore.
  - For each energy it calls the function `compute_resonances_per_energy` (see below), and it outputs 2 lists: a list containing the $a_{eff}$ values and the computed ressonant energy (this lists are as long as the number of dispersors).
  - Those results are dumped in a general array that stores everything.
  - Then, we mask the elements of those arrays that have an imaginary part of the ressonance smaller than $10^{-6}$, we take the logarithm and we make a 2D histogram, i.e. for some interval of real part of the ressonance, and effective scattering length, we count how many states have been found.

### compute_resonances_per_energy

We will use that, in reduced units, $E=\frac{k^2}{2}$. Then this function generates a set of dispersors (it builds a square lattice and then populates each site with a small probability, default p=0.1). Then, the program calls the function `distance_between_dispersors()` which fills an upper triangular matrix with the distance between dispersors. Then it computes the $M^{\infty}$ with the `M_inf()` function. After this, the `resonances()` function computes the Newton step for each eigenvalue of the matrix.

### Resonances function

The function first diagonalizes the $M^{\infty}$ matrix, and then calls the function `resonance()`, which actually makes the computation.

#### Resonance

This function selects the eigenvalue and eigenvector, computes the $a_{eff}$ and calls the `newton_step()` function.

##### Newton_step

This function calls the function that computes the derivative, and performs the step proposed.
