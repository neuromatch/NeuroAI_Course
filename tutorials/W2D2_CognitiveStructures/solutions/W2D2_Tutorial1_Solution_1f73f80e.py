
def non_separable(x):
    """Compute non-separable function for given array of 2-dimenstional vectors.

    Inputs:
    - x (np.ndarray of shape (n, 2)): n 2-dimensional vectors.

    Outputs:
    - y (np.ndarray of shape (n, 1)): non-separable function value for each of the vectors.
    """
    return np.sin(np.multiply(x[:,0],x[:,1]))