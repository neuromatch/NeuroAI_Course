T_ar = np.arange(len(sig))

#100 different frequency values from 0.001 to 1, then apply each frequency on `T_ar`
freqs = np.linspace(0.001, 1, 100)
set_sigs = [np.sin(T_ar*f) for f in freqs]

# define 'reg' --- an sklearn object of OrthogonalMatchingPursuit, and fit it to the data, where the frequency bases are the features and the signal is the label
reg = OrthogonalMatchingPursuit(fit_intercept = True, n_nonzero_coefs = 10).fit(np.vstack(set_sigs).T, sig)