
sims = []

for obj_idx, obj in enumerate(obj_names):
    sims.append(query_ssps @ ssps[obj].flatten())

plt.figure(figsize=(8, 2.4))
plot_2d_similarity(sims, obj_names, (dim0.size, dim1.size))