
num_iters = 500
losses = []
sims = []
lr = 1e-1
ant_names = ["blue", "odd"]
cons_names = ["even", "green"]

transform = np.zeros((1,encoder.ssp_dim))
for i in range(num_iters):
    loss = 0
    for rule, ant_name, cons_name in zip(rules, ant_names, cons_names):

        #perfect similarity
        y_true = np.eye(len(action_names))[action_names.index(ant_name),:] + np.eye(len(action_names))[4+action_names.index(cons_name),:]

        #prediction with current transform (a_hat = transform * rule)
        a_hat = sspspace.SSP(transform) * rule

        #similarity with current transform
        sim_mat = action_space @ a_hat.T

        #cleanup
        y_hat = softmax(sim_mat)

        #true solution (a* = ant_name + not * cons_name)
        a_true = (vocab[ant_name] + vocab['not']*vocab[cons_name]).normalize()

        #calculate loss
        loss += log_loss(y_true, y_hat)

        #update transform (T <- T - lr * (T - A* * (~rule)))
        transform -= (lr) * (transform - np.array(a_true * ~rule))
        transform = transform / np.linalg.norm(transform)

        #save predicted similarities if it is last iteration
        if i == num_iters - 1:
            sims.append(sim_mat)

    #save loss
    losses.append(np.copy(loss))

plt.figure(figsize=(15,5))
plot_training_and_choice(losses, sims, ant_names, cons_names, action_names)