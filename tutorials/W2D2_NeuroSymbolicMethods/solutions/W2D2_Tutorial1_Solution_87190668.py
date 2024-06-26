
"""
Discussion: How would you explain the lines `sims = vector @ phis.T` in the previous coding exercises?

We compute the similarity of `vector` to all the other references `phi` using the dot product. `vector` has shape `d` and phis has shape `m x d`, where `m` is the number of references. This yields `m` similarity values, one for each reference.
""";