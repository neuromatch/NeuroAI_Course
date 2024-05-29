
def dcl_loss(pos_pairs, neg_pairs, indices_tuple, temperature=0.07):
    # This is the modified InfoNCE loss called "Decoupled Contrastive Learning" for small batch sizes
    # Basically You remove the numerator from the sum to the denominator

    a1, p, a2, _ = indices_tuple  # Unpack indices

    if len(a1) > 0 and len(a2) > 0:
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
    return 0

def pair_based_loss(mat, indices_tuple, lossfunc):
    # Computes pair-based loss using the provided loss function
    a1, p, a2, n = indices_tuple  # Unpack indices
    pos_pair, neg_pair = [], []
    if len(a1) > 0:
        pos_pair = mat[a1, p]  # Extract positive pairs
    if len(a2) > 0:
        neg_pair = mat[a2, n]  # Extract negative pairs
    return lossfunc(pos_pair, neg_pair, indices_tuple)  # Apply loss function