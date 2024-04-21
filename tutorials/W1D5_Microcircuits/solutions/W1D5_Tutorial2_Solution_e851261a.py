non_linearities = {
    'ReLU': nn.ReLU(),
    'ReLU6': nn.ReLU6(),
    'SoftPlus': nn.Softplus(beta=1, threshold=20),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh()
}