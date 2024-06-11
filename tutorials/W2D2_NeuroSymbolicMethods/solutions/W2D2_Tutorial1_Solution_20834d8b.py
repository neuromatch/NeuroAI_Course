
five_unbind_two = sspspace.SSP(integers[4]) * ~sspspace.SSP(integers[1])
sims = np.einsum('nd,md->nm', five_unbind_two, integers)