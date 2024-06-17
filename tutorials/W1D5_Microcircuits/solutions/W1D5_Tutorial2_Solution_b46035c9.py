class ReLUNet(nn.Module):
  """
  ReLUNet architecture
  Structure is as follows:
  y = Σi(ai * ReLU(θi - x))
  """
  # Define the structure of your network
  def __init__(self, n_units):
    """
    Args:
      n_units (int): Number of hidden units

    Returns:
      Nothing
    """
    super(ReLUNet, self).__init__()
    # Create input thresholds
    self.input_threshold_weights = nn.Parameter(torch.abs(torch.randn(n_units)))
    self.non_linearity = nn.ReLU()
    self.output_layer = nn.Linear(n_units, 1)
    nn.init.xavier_normal_(self.output_layer.weight)

  def forward(self, x):
    """
    Args:
      x: torch.Tensor
        Input tensor of size ([1])
    """
    op = self.input_threshold_weights - x #prepare the input to be passed through ReLU
    op = self.non_linearity(op) #apply ReLU
    op = self.output_layer(op) #run through output layer
    return op

  # Choose the most likely label predicted by the network
  def predict(self, x):
    """
    Args:
      x: torch.Tensor
        Input tensor of size ([1])
    """
    output = self.forward(x)
    return output