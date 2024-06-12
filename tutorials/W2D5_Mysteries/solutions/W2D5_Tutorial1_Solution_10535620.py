
# Experiment parameters
mu = np.array([[3.5, 0.5], [0.5, 3.5], [0.5, 0.5]])
Nsubjects = 30
Ntrials = 600
cond = np.concatenate((np.ones(Ntrials//3), np.ones(Ntrials//3)*2, np.ones(Ntrials//3)*3))
Wprior = [0.5, 0.5]
Aprior = 0.5

# Sensory precision values
gamma = np.linspace(0.1, 10, 6)

# Initialize lists for results
all_KL_w_yes = []
sem_KL_w_yes = []
all_KL_w_no = []
sem_KL_w_no = []
all_KL_A_yes = []
sem_KL_A_yes = []
all_KL_A_no = []
sem_KL_A_no = []
all_prob_y = []

for y in tqdm(gamma, desc='Processing gammas'):
    Sigma = np.diag([1./np.sqrt(y)]*2)
    mean_KL_w = np.zeros((Nsubjects, 4))
    mean_KL_A = np.zeros((Nsubjects, 4))
    prob_y = np.zeros(Nsubjects)

    for s in tqdm(range(Nsubjects), desc=f'Subjects for gamma={y}', leave=False):
        KL_w = np.zeros(len(cond))
        KL_A = np.zeros(len(cond))
        posteriorAware = np.zeros(len(cond))

        # Generate sensory samples
        X = np.array([multivariate_normal.rvs(mean=mu[int(c)-1, :], cov=Sigma) for c in cond])

        # Model inversion for each trial
        for i, x in enumerate(X):
            post_w, post_A, KL_w[i], KL_A[i] = HOSS_evaluate(x, mu, Sigma, Aprior, Wprior)
            posteriorAware[i] = post_A[1]  # Assuming post_A is a tuple with awareness probability at index 1

        binaryAware = posteriorAware > 0.5
        for i in range(4):
            conditions = [(cond == 3), (cond != 3), (cond != 3), (cond == 3)]
            aware_conditions = [(binaryAware == 0), (binaryAware == 0), (binaryAware == 1), (binaryAware == 1)]
            mean_KL_w[s, i] = np.mean(KL_w[np.logical_and(aware_conditions[i], conditions[i])])
            mean_KL_A[s, i] = np.mean(KL_A[np.logical_and(aware_conditions[i], conditions[i])])

        prob_y[s] = np.mean(binaryAware[cond != 3])

    # Aggregate results across subjects
    all_KL_w_yes.append(np.mean(mean_KL_w[:, 2:4].flatten()))
    sem_KL_w_yes.append(np.std(mean_KL_w[:, 2:4].flatten()) / np.sqrt(Nsubjects))
    all_KL_w_no.append(np.mean(mean_KL_w[:, :2].flatten()))
    sem_KL_w_no.append(np.std(mean_KL_w[:, :2].flatten()) / np.sqrt(Nsubjects))
    all_KL_A_yes.append(np.mean(mean_KL_A[:, 2:4].flatten()))
    sem_KL_A_yes.append(np.std(mean_KL_A[:, 2:4].flatten()) / np.sqrt(Nsubjects))
    all_KL_A_no.append(np.mean(mean_KL_A[:, :2].flatten()))
    sem_KL_A_no.append(np.std(mean_KL_A[:, :2].flatten()) / np.sqrt(Nsubjects))
    all_prob_y.append(np.mean(prob_y))

with plt.xkcd():

    # Create figure
    plt.figure(figsize=(16, 4.67))

    # First subplot: Probability of reporting "seen" for w_1 or w_2
    plt.subplot(1, 3, 1)
    plt.plot(gamma, all_prob_y, linewidth=2)
    plt.xlabel('SOA (precision)')
    plt.ylabel('Prob. report "seen" for w_1 or w_2')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.box(False)

    # Second subplot: K-L divergence, perceptual states
    plt.subplot(1, 3, 2)
    plt.errorbar(gamma, all_KL_w_yes, yerr=sem_KL_w_yes, linewidth=2, label='Seen')
    plt.errorbar(gamma, all_KL_w_no, yerr=sem_KL_w_no, linewidth=2, label='Unseen')
    plt.legend(frameon=False)
    plt.xlabel('SOA (precision)')
    plt.ylabel('K-L divergence, perceptual states')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.box(False)

    # Third subplot: K-L divergence, awareness state
    plt.subplot(1, 3, 3)
    plt.errorbar(gamma, all_KL_A_yes, yerr=sem_KL_A_yes, linewidth=2, label='Seen')
    plt.errorbar(gamma, all_KL_A_no, yerr=sem_KL_A_no, linewidth=2, label='Unseen')
    plt.legend(frameon=False)
    plt.xlabel('SOA (precision)')
    plt.ylabel('K-L divergence, awareness state')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.box(False)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()