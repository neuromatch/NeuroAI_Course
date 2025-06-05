
set_seed(42)

vector_length = 1024
symbol_names = ['CIRCLE','SQUARE','TRIANGLE']

vocab = make_vocabulary(vector_length)
vocab.populate(';'.join(symbol_names))
print(list(vocab.keys()))

circle = vocab['CIRCLE']
square = vocab['SQUARE']
triangle = vocab['TRIANGLE']

print('|circle| =', np.linalg.norm(circle.v))
print('|triangle| =', np.linalg.norm(square.v))
print('|square| =', np.linalg.norm(triangle.v))