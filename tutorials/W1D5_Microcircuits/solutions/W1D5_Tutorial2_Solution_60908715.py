def HardTanh(x):
  """
  Calculates `tanh` output for the given input data.

  Inputs:
  - x (np.ndarray): input data.

  Outputs:
  - output (np.ndarray): `tanh(x)`.
  """
  min_val = -1
  max_val = 1
  output = np.copy(x)
  output[output>max_val] = max_val
  output[output<min_val] = min_val
  return output

def LeakyHardTanh(x, leak_slope=0.03):
  """
  Calculate `tanh` output for the given input data with the leaky term.

  Inputs:
  - x (np.ndarray): input data.
  - leak_slope (float, default = 0.03): leaky term.

  Outputs:
  - output (np.ndarray): `tanh(x)`.
  """
  output = np.copy(x)
  output = HardTanh(output) + leak_slope*output
  return output

def InverseLeakyHardTanh(y, leak_slope=0.03):
  """
  Calculate input into the `tanh` function with the leaky term for the given output.

  Inputs:
  - y (np.array): output of leaky tanh function.
  - leak_slope (float, default = 0.03): leaky term.

  Outputs:
  - output (np.array): input into leaky tanh function.
  """
  ycopy = np.copy(y)
  output = np.where(
      np.abs(ycopy) >= 1+leak_slope, \
      (ycopy - np.sign(ycopy))/leak_slope, \
      ycopy/(1+leak_slope)
  )
  return output