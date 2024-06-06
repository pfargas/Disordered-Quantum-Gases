from numba import njit, prange

@njit(parallel=True)
def distances_between_dispersors(distances_matrix, dispersor_set):
    """
    Calculate the distances between all pairs of dispersors in the dispersor set.
    This function only computes the upper triangular part of the matrix, as the matrix is symmetric.
    It is up to the user to ensure the final matrix is added to its transpose to obtain the full matrix.
    Inputs:
    distances_matrix: numpy array of shape (n, n) where n is the number of dispersors in the set. Should be initialized with zeros.
    dispersor_set: numpy array of shape (n, 2) where n is the number of dispersors in the set. Each row contains the x and y coordinates of a dispersor.
    Outputs:
    distances_matrix: numpy array of shape (n, n) where n is the number of dispersors in the set. Contains the distances between all pairs of dispersors.
    """
    n = len(dispersor_set)
    for i in prange(n-1):
        for j in prange(i+1,n):
            distances_matrix[i, j] = ((dispersor_set[i, 0] - dispersor_set[j, 0])**2 + (dispersor_set[i, 1] - dispersor_set[j, 1])**2)**0.5
    return distances_matrix
