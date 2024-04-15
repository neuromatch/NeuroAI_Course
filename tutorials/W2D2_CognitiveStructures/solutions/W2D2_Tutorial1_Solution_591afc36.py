
set_seed(42)

ssp_space = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=1024)
bound_phis = ssp_space.encode(xs)

ssp_space0 = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=1024)
ssp_space1 = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=1024)
bundle_phis = ssp_space0.encode(xs[:,0][:,None]) + ssp_space1.encode(xs[:,1][:,None]) #remember that input to `encode` should be 2-dimensional, thus we need to create extra dimension