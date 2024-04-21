T_ar = np.arange(len(sig))

# create a set of 20 signals with different frequencies, from 1 to 101, using sin(2\pi f t).
freqs = np.linspace(0.001,1, 100)
set_sigs = [np.sin(f*T_ar) for f in freqs]

# define 'reg' --- an sklearn object of OrthogonalMatchingPursuit and fit it to the data, where the frequency bases are the atoms and the signal is the label
reg = OrthogonalMatchingPursuit(fit_intercept = True, n_nonzero_coefs = 10).fit(np.vstack(set_sigs).T, sig)