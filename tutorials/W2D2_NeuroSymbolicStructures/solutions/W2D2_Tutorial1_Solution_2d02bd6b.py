
new_rule = (vocab['ant'] * vocab['red'] + vocab['relation'] * vocab['implies'] + vocab['cons'] * vocab['prime']).normalize()

a_hat = sspspace.SSP(transform) * new_rule

new_sims = np.einsum('nd,md->nm', action_space, a_hat)
y_hat = softmax(new_sims)