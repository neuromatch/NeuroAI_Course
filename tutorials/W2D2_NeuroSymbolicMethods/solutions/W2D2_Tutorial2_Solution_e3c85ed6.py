
set_seed(42)

new_symbol_names = ['dollar', 'peso', 'ottawa', 'mexico-city', 'currency', 'capital']
new_discrete_space = sspspace.DiscreteSPSpace(new_symbol_names, ssp_dim=1024, optimize=False)

new_objs = {n:new_discrete_space.encode(n) for n in new_symbol_names}

cleanup = sspspace.Cleanup(new_objs)

new_objs['canada'] = ((new_objs['currency'] * new_objs['dollar']) + (new_objs['capital'] * new_objs['ottawa'])).normalize()
new_objs['mexico'] = ((new_objs['currency'] * new_objs['peso']) + (new_objs['capital'] * new_objs['mexico-city'])).normalize()