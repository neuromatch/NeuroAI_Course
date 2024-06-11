
set_seed(42)

symbol_names = ['dollar','peso', 'ottawa','mexico-city','currency','capital']
discrete_space = sspspace.DiscreteSPSpace(symbol_names, ssp_dim=1024, optimize=False)

objs = {n:discrete_space.encode(n) for n in symbol_names}

cleanup = sspspace.Cleanup(objs)

objs['canada'] = (objs['currency'] * objs['dollar'] + objs['capital'] * objs['ottawa']).normalize()
objs['mexico'] = (objs['currency'] * objs['peso'] + objs['capital'] * objs['mexico-city']).normalize()