def gained_dot_product_attention(x: torch.Tensor,  # input vector
                                 q_1: torch.Tensor,  # query vector 1
                                 q_2: torch.Tensor,  # query vector 2
                                 z_1: float,  # query gain 1
                                 z_2: float,  # query gain 2
                                 ):
    """This function computes the gained dot product attention
    Args:
        x (Tensor): input vector
        q_1 (Tensor): query vector 1
        q_2 (Tensor): query vector 2
        z_1 (float): query gain 1
        z_2 (float): query gain 2
    Returns:
        w (Tensor): attention weights
        y (float): gained dot product attention
    """
    w = torch.softmax(z_1 * q_1 + z_2 * q_2, dim=0)
    y = torch.dot(w, x)
    return w, y