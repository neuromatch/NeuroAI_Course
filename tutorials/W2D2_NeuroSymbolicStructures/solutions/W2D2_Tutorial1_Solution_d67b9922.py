
set_seed(42)
encoder = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=1024)

xs = np.linspace(-4,4,401)[:,None] #we expect the encoded values to be two-dimensional in `encoder.encode()` so we add extra dimension
phis = encoder.encode(xs)

sims = np.einsum('d,md->m', phis[200, :], phis)