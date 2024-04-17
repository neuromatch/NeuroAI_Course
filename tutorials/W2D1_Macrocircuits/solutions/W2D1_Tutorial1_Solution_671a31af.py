set_seed(42)

def make_data(net, n_in, n_examples):
    """
    Generate data by sampling from a multivariate gaussian distribution, and output data by passing the inputs through the network.

    Inputs:
    - net (nn.Sequential): network.
    - n_in (int): input dimension.
    - n_examples (int): number of data examples to generate.

    Outputs:
    - X (torch.tensor): input data.
    - y (torch.tensor): output data.
    """
    X = torch.randn(n_examples, n_in)
    y = net(X).detach()
    return X, y

X, y = make_data(net, 10, 10000000)
X[0][0]