
objs['queen_query'] = (objs['king'] * ~objs['male']) * objs['female']

object_names = list(objs.keys())
sims = np.zeros((len(object_names), len(object_names)))

for name_idx, name in enumerate(object_names):
    for other_idx in range(name_idx, len(object_names)):
        sims[name_idx, other_idx] = sims[other_idx, name_idx] = (objs[name] | objs[object_names[other_idx]]).item()

plot_similarity_matrix(sims, object_names, values = True)