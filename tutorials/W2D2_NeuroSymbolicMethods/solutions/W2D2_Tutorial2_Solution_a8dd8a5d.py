
new_rule = (vocab['ant'] * vocab['red'] + vocab['relation'] * vocab['implies'] + vocab['cons'] * vocab['prime']).normalize()

#apply transform on new rule to test the generalization of the transform
a_hat = sspspace.SSP(transform) * new_rule

new_sims = action_space @ a_hat.T
y_hat = softmax(new_sims)

plot_choice([new_sims], ["red"], ["prime"], action_names)