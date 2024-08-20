
new_objs['peso_query'] = cleanup(~new_objs['canada'] * new_objs['dollar'] * new_objs['mexico'])

object_names = list(new_objs.keys())
sims = np.zeros((len(object_names), len(object_names)))

for name_idx, name in enumerate(object_names):
    for other_idx in range(name_idx, len(object_names)):
        sims[name_idx, other_idx] = sims[other_idx, name_idx] = (new_objs[name] | new_objs[object_names[other_idx]]).item()

plot_similarity_matrix(sims, object_names)