
five_unbind_two = sspspace.SSP(integers[4]) * ~sspspace.SSP(integers[1])
five_unbind_two_sims = np.einsum('nd,md->nm', five_unbind_two, integers)