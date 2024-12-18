
set_seed(42)
new_encoder = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=1024)

xs = np.linspace(-4,4,401)[:,None] #we expect the encoded values to be two-dimensional in `encoder.encode()` so we add extra dimension
phis = new_encoder.encode(xs)

#`0` element is right in the middle of phis array! notice that we have 401 samples inside it
real_line_sims = phis[200, :] @ phis.T