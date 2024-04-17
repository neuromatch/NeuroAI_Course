def make_MLP(n_in, W, D, nonlin='tanh'):
    """
    Create `nn.Sequnetial()` fully-connected model in pytorch with the given parameters.

    Inputs:
    - n_in (int): input dimension.
    - W (int): width of the network.
    - D (int): depth if the network.
    - nonlin (str, default = "tanh"): activation function to use.

    Outputs:
    - net (nn.Sequential): network.
    """

    #activation function
    if nonlin == 'tanh':
        nonlin = nn.Tanh()
    elif nonlin == 'relu':
        nonlin == nn.ReLU()
    else:
        assert(False)

    # Assemble D-1 hidden layers and one output layer

    #input layer
    layers = [nn.Linear(n_in, W, bias = False), nonlin]
    for i in range(D - 2):
        #linear layer
        layers.append(nn.Linear(W, W, bias = False))
        #activation function
        layers.append(nonlin)
    #output layer
    layers.append(nn.Linear(W, 1, bias = False))

    return nn.Sequential(*layers)

net = make_MLP(10, 3, 2)