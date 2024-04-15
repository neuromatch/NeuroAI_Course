
set_seed(42)

vector_length = 1024
symbol_names = ['circle','square','triangle']
discrete_space = sspspace.DiscreteSPSpace(symbol_names, ssp_dim=vector_length, optimize = False)

circle = discrete_space.encode('circle')
square = discrete_space.encode('square')
triangle = discrete_space.encode('triangle')

print('|circle| =', np.linalg.norm(circle))
print('|triangle| =', np.linalg.norm(triangle))
print('|square| =', np.linalg.norm(square))