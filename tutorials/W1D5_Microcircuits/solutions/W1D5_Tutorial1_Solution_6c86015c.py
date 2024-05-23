# define a list or numpy array of optional cardinallities from 1 to 51 in intervals of 5.
cardinalities = np.arange(1,101,5)

# for each of the optional cardinalities, run OMP using the pixel's signal from before. Create a list called "regs" that include all OMP's fitted objects
regs = [OrthogonalMatchingPursuit(fit_intercept = True, n_nonzero_coefs = card).fit(np.vstack(set_sigs).T, sig) for card in cardinalities]