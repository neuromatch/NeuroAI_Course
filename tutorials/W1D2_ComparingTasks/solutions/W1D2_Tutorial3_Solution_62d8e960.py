t = np.arange(1,101) # Array representing trials from 1 to 100
p_L = 0.25
p_R = 0.75

# In the best case scenario, the agent chooses the best arm every trial,
# leading to a cumulative regret of 0.
cr_best = np.zeros(100)

# In the worst-case scenario, the agent chooses the worst arm every trial,
# leading to per trial regret of the best arm's reward - the worst arm's reward
per_trial_regret = p_R - p_L
regret_worst = per_trial_regret * np.ones(100)
cr_worst = np.cumsum(regret_worst)

with plt.xkcd():
    plt.plot(t, cr_best, label = 'best case')
    plt.plot(t, cr_worst, label = 'worst case')

    plt.xlabel('trial')
    plt.ylabel('cumulative regret')
    plt.legend()