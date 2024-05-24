set_seed(42)

#define axis vector
axis_vectors = ['one']

encoder = sspspace.DiscreteSPSpace(axis_vectors, ssp_dim=1024, optimize=False)

#vocabulary
vocab = {w:encoder.encode(w) for w in axis_vectors}

#we will add new vectors to this list
integers = [vocab['one']]

max_int = 5
for i in range(2, max_int + 1):
    #bind one more "one" to the previous integer to get the new one
    integers.append(integers[-1] * vocab['one'])

integers = np.array(integers).squeeze()
integers_sims = np.einsum('nd,md->nm',integers,integers)