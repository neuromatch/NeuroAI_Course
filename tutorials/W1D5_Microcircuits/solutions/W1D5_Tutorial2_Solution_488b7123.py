def normalize(x, sigma, p, g):
  """
  Inputs:
  - x(np.ndarray): Input array (n_samples * n_dim)
  - sigma(float): Smoothing factor
  - p(int): p-norm
  - g(int): scaling factor

  Outputs:
  - xnorm (np.ndarray): normalized values.
  """
  # Raise the absolute value of x to the power p
  xp = np.power(np.abs(x), p)
  # Sum the x over the dimensions (n_dim) axis
  xp_sum = np.sum(np.power(np.abs(x), p), axis=1)
  # Correct the dimensions of xp_sum and taking average reduces the dimensions
  # Making xp_sum a row vector of shape (1, n_dim)
  xp_sum = np.expand_dims(xp_sum, axis=1)
  # Raise the sum to the power 1/p and add smoothing factor (sigma)
  denominator = sigma + np.power(xp_sum, 1/p)
  # Scale the input data with a factor of g
  numerator = x*g
  # Calculate normalized x
  xnorm = numerator/denominator
  return xnorm