set_seed(42)

def initialize_layers(net,sigma):
    """
    Set weight to each of the parameters in the model of value sigma/sqrt(n_in), where n_in is the number of inputs to the layer.

    Inputs:
    - net (nn.Sequential): network.
    - sigma (float): standard deviation.
    """
    for param in net.parameters():
        n_in = param.shape[1]
        nn.init.normal_(param, std = sigma/np.sqrt(n_in))

initialize_layers(net, 1)
next(net.parameters())[0][0]