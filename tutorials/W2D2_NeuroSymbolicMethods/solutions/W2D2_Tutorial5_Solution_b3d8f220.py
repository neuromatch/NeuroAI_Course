
#objects are located in `objs` and positions in `ssps`
bound_objects = [objs[n] * ssps[n] for n in obj_names]

sims = []

for obj_idx, obj in enumerate(obj_names):
    sims.append(query_ssps @ bound_objects[obj_idx].flatten())

plt.figure(figsize=(8, 2.4))
plot_2d_similarity(sims, obj_names, (dim0.size, dim1.size))