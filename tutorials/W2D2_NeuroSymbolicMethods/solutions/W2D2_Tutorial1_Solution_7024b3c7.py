
set_seed(42)

symbol_names = ['fire-fighter','math-teacher','sales-manager']
discrete_space = sspspace.DiscreteSPSpace(symbol_names, ssp_dim=1024, optimize=False)

vocab = {n:discrete_space.encode(n) for n in symbol_names}

noisy_vector = 0.2 * vocab['fire-fighter'] + 0.15 * vocab['math-teacher'] + 0.3 * vocab['sales-manager']

sims = np.array([noisy_vector | vocab[name] for name in symbol_names]).squeeze()

plot_line_similarity_matrix(sims, symbol_names, multiple_objects = False, title = 'Similarity - pre cleanup')