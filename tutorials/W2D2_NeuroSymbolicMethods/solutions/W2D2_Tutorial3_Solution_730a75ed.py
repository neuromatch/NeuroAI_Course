
def non_separable(x):
    """Compute non-separable function for given array of 2-dimenstional vectors.

    Inputs:
    - x (np.ndarray of shape (n, 2)): n 2-dimensional vectors.

    Outputs:
    - y (np.ndarray of shape (n, 1)): non-separable function value for each of the vectors.
    """
    return np.sin(np.multiply(x[:, 0], x[:, 1]))

x0_non_separable = np.linspace(-4, 4, 100)
X_non_separable, Y_non_separable = np.meshgrid(x0_non_separable,x0_non_separable)
xs_non_separable = np.vstack((X_non_separable.flatten(), Y_non_separable.flatten())).T

ys_non_separable = non_separable(xs_non_separable)