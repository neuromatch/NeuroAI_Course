
set_seed(42)

class Cleanup:
    def __init__(self, vocab, temperature=1e5):
        self.weights = np.array([vocab[k] for k in vocab.keys()]).squeeze()
        self.temp = temperature
    def __call__(self, x):
        sims = x @ self.weights.T
        max_sim = softmax(sims * self.temp, axis=0)
        return sspspace.SSP(max_sim @ self.weights)


cleanup = Cleanup(vocab)

clean_vector = cleanup(noisy_vector)

clean_sims = np.array([clean_vector | vocab[name] for name in symbol_names]).squeeze()

plot_double_line_similarity_matrix([sims, clean_sims], symbol_names, ['Noisy Similarity', 'Clean Similarity'], title = 'Similarity - post cleanup')