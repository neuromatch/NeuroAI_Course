def generic_function(x, seed):
  """Google style doc string. Brief summary of what function does here

  Args:
    x (ndarray): An array of shape (N,) that contains blah blah
    seed (integer): random seed for reproducibility

  Returns:
    ndarray: The output is blah blah
  """

  # Have a comment for every line of code they need to write, and when possible have
  # variables written with ellipses where they should fill in or ellipses where they should
  # fill in inputs to functions
  y = multiply_array(x, 5, seed)

  # Another comment because they need to add another line of code
  z = y + 6

  return z


x = np.array([4, 5, 6])

# We usually define the plotting function in the hidden Helper Functions
# so students don't have to see a bunch of boilerplate matplotlib code
## Uncomment the code below to test your function

# z = generic_function(x, seed=2021)
# with plt.xkcd():
#   plotting_z(z)