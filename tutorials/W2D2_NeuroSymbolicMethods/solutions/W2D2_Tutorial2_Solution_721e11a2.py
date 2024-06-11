
set_seed(42)

symbol_names = ['monarch','heir','male','female']
discrete_space = sspspace.DiscreteSPSpace(symbol_names, ssp_dim=1024, optimize=False)

objs = {n:discrete_space.encode(n) for n in symbol_names}

objs['king'] = objs['monarch'] * objs['male']
objs['queen'] = objs['monarch'] * objs['female']
objs['prince'] = objs['heir'] * objs['male']
objs['princess'] = objs['heir'] * objs['female']