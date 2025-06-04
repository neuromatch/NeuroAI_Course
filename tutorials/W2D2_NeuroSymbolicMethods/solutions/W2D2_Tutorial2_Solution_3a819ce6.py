
new_rule = (vocab['ANT'] * vocab['RED'] + vocab['RELATION'] * vocab['IMPLIES'] + vocab['CONS'] * vocab['PRIME']).normalized()

#apply transform on new rule to test the generalization of the transform
a_hat = spa.SemanticPointer(transform) * new_rule

new_sims = np.einsum('nd,d->n', action_space, a_hat.v)
y_hat = softmax(new_sims)