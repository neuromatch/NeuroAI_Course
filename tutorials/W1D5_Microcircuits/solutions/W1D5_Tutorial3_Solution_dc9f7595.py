def scaled_dot_product_attention(Q, K, V):
    """ Scaled dot product attention
    Args:
        Q: queries (B, H, d, n)
        K: keys (B, H, d, n)
        V: values (B, H, d, n)
    Returns:
        Attention tensor (B, H, d, n), Scores (B, H, d, d)
    Notes:
        (B, H, d, n): batch size, H: number of heads, d: key-query dim, n: embedding dim
    """

    assert K.shape == Q.shape and K.shape == V.shape, "Queries, Keys and Values must have the same shape"
    B, H, d, n = K.shape  # batch_size, num_heads, key-query dim, embedding dim
    scale = math.sqrt(d)

    # # START using torch.bmm #######################
    Q_ = Q.view(B * H, d, n) # necessary only if using torch.bmm
    K_ = K.view(B * H, d, n) # necessary only if using torch.bmm
    V_ = V.view(B * H, d, n) # necessary only if using torch.bmm
    Q_mm_K = torch.bmm(Q_, K_.transpose(1, 2))  # dot-product reducing the n dimension
    S = Q_mm_K / scale  # score or scaled dot product
    S_sm = torch.softmax(S, dim=2)  # softmax
    A = torch.bmm(S_sm, V_)  # Attention
    S = S.view(B, H, d, d) # necessary only if using torch.bmm
    A = A.view(B, H, d, n) # necessary only if using torch.bmm
    # # END using torch.bmm #########################
    # # START using torch.einsum ####################
    Q_mm_K = torch.einsum("bhdn,bhen->bhde", Q, K)  # dot-product reducing the n dimension
    S = Q_mm_K / scale  # score or scaled dot product
    S_sm = torch.softmax(S, dim=-1)  # softmax
    A = torch.einsum("bhde,bhen->bhdn", S_sm, V)  # Attention
    # # END using torch.einsum ######################

    assert S.shape == (B, H, d, d), "Score tensor does not have the correct shape"
    assert A.shape == (B, H, d, n), "Attention tensor does not have the correct shape"
    return A, S  # Attention, Score