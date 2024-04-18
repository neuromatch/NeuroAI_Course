
def get_num_params(n_in,W,D):
    """
    Simple function to compute number of learned parameters in an MLP with given dimensions.

    Inputs:
    - n_in (int): input dimension.
    - W (int): width of the network.
    - D (int): depth if the network.

    Outputs:
    - num_params (int): number of parameters in the network.
    """
    input_params = n_in * W
    hidden_layers_params = (D-2) * W**2
    output_params = W
    return input_params + hidden_layers_params + output_params

get_num_params(10, 3, 2)