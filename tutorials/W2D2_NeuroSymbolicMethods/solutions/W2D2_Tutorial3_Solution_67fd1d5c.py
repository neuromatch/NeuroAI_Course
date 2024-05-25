
def rastrigin(x):
    """Compute Rastrigin function for given array of d-dimenstional vectors.

    Inputs:
    - x (np.ndarray of shape (n, d)): n d-dimensional vectors.

    Outputs:
    - y (np.ndarray of shape (n, 1)): Rastrigin function value for each of the vectors.
    """
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2*np.pi*x), axis=1)