
def HOSS_evaluate(X, mu, Sigma, Aprior, Wprior):
    """
    Inference on 2D Bayes net for asymmetric inference on presence vs. absence.
    """

    # Initialise variables and conditional prob tables
    p_A = np.array([1 - Aprior, Aprior])  # prior on awareness state A
    p_W_a1 = np.append(Wprior, 0)  # likelihood of world states W given aware, last entry is absence
    p_W_a0 = np.append(np.zeros(len(Wprior)), 1)  # likelihood of world states W given unaware, last entry is absence
    p_W = (p_W_a1 + p_W_a0) / 2  # prior on W marginalising over A (for KL)

    # Compute likelihood of observed X for each possible W (P(X|mu_w, Sigma))
    lik_X_W = np.array([multivariate_normal.pdf(X, mean=mu_i, cov=Sigma) for mu_i in mu])
    p_X_W = lik_X_W / lik_X_W.sum()  # normalise to get P(X|W)

    # Combine with likelihood of each world state w given awareness state A
    lik_W_A = np.vstack((p_X_W * p_W_a0 * p_A[0], p_X_W * p_W_a1 * p_A[1]))
    post_A = lik_W_A.sum(axis=1)  # sum over W
    post_A = post_A / post_A.sum()  # normalise

    # Posterior over W (P(W|X=x) marginalising over A)
    post_W = lik_W_A.sum(axis=0)  # sum over A
    post_W = post_W / post_W.sum()  # normalise

    # KL divergences
    KL_W = (post_W * np.log(post_W / p_W)).sum()
    KL_A = (post_A * np.log(post_A / p_A)).sum()

    return post_W, post_A, KL_W, KL_A