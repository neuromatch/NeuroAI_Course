
def dcl_loss(pos_pairs, neg_pairs, indices_tuple, temperature=0.07):
    """
    Computes the Decoupled Contrastive Learning loss.

    Returns:
        torch.Tensor: The computed loss value.
    """
    a1, _, a2, _ = indices_tuple  # Unpack indices

    if len(a1) == 0 or len(a2) == 0:
        return 0

    dtype = neg_pairs.dtype
    pos_pairs = pos_pairs.unsqueeze(1) / temperature  # Scale positive pairs by temperature
    neg_pairs = neg_pairs / temperature  # Scale negative pairs by temperature
    n_per_p = to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)  # Indicator matrix for matching pairs
    neg_pairs = neg_pairs * n_per_p  # Zero out non-matching pairs
    neg_pairs[n_per_p == 0] = neg_inf(dtype)  # Replace non-matching pairs with negative infinity

    # Compute the maximum value for numerical stability
    max_val = torch.max(
        pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
    ).detach()
    # Compute numerator and denominator for the loss
    numerator = torch.exp(pos_pairs - max_val).squeeze(1)
    denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1)
    log_exp = torch.log((numerator / denominator) + small_val(dtype))
    return -log_exp  # Return the negative log of the exponential

def pair_based_loss(similarities, indices_tuple, lossfunc):
    """
    Computes pair-based loss using the provided loss function.

    Args:
        similarities : torch.Tensor
            A tensor of pairwise similarities. For n_examples, the shape should be
            (n_examples, n_examples).
        indices_tuple : tuple
            A tuple of indices for positive and negative pairs. The tuple should
            contain 4 tensors: a1, p, a2, n. The tensors a1 and p contain indices
            for positive pairs, while a2 and n contain indices for negative pairs.
            a1 and p should have the same length, and a2 and n should have the same
            length. a1[i] and p[i] should form a positive pair, such that they have the
            same label. Similarly, a2[i] and n[i] should form a negative pair, such that
            they have different labels.
        lossfunc : function
            The loss function to be applied for computing the loss.
    """
    # Computes pair-based loss using the provided loss function
    a1, p, a2, n = indices_tuple  # Unpack indices
    pos_pair, neg_pair = [], []
    if len(a1) > 0:
        pos_pair = similarities[a1, p]  # Extract positive pairs
    if len(a2) > 0:
        neg_pair = similarities[a2, n]  # Extract negative pairs
    return lossfunc(pos_pair, neg_pair, indices_tuple)  # Apply loss function