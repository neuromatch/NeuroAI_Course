
num_iters = 500
losses = []
sims = []
lr = 1e-1
ant_names = ["BLUE", "ODD"]
cons_names = ["EVEN", "GREEN"]
vector_length = 1024

transform = np.zeros((vector_length))
for i in range(num_iters):
    loss = 0
    for rule, ant_name, cons_name in zip(rules, ant_names, cons_names):

        #perfect similarity
        y_true = np.eye(len(action_names))[action_names.index(ant_name),:] + np.eye(len(action_names))[4+action_names.index(cons_name),:]

        #prediction with current transform (a_hat = transform * rule)
        a_hat = spa.SemanticPointer(transform) * rule

        #similarity with current transform
        sim_mat = np.einsum('nd,d->n', action_space, a_hat.v)

        #cleanup
        y_hat = softmax(sim_mat)

        #true solution (a* = ant_name + not * cons_name)
        a_true = (vocab[ant_name] + vocab['NOT']*vocab[cons_name]).normalized()

        #calculate loss
        loss += log_loss(y_true, y_hat)

        #update transform (T <- T - lr * (A* * (~rule)))
        transform -= (lr) * (transform - (a_true * ~rule).v)
        transform = transform / np.linalg.norm(transform)

        #save predicted similarities if it is last iteration
        if i == num_iters - 1:
            sims.append(sim_mat)

    #save loss
    losses.append(np.copy(loss))