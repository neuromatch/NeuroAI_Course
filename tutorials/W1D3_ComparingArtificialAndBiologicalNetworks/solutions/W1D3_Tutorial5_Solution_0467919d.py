
n_delays = 20
delay_interval = 10

models = [trajectory_circle, trajectory_oval, trajectory_walk]
dsa = DSA(models, n_delays=n_delays, delay_interval=delay_interval)
similarities = dsa.fit_score()

labels = ['Circle', 'Oval', 'Walk']
ax = sns.heatmap(similarities, xticklabels=labels, yticklabels=labels)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('DSA Score');
plt.title("Dynamic Similarity Analysis Score among Trajectories");