new_object_names = ['red','red^','red*circle','circle','circle^']
new_objs = objs

new_objs['red^'] = new_objs['red*circle'] * ~new_objs['circle']
new_objs['circle^'] = new_objs['red*circle'] * ~new_objs['red']

new_obj_sims = np.zeros((len(new_object_names), len(new_object_names)))

for name_idx, name in enumerate(new_object_names):
    for other_idx in range(name_idx, len(new_object_names)):
        new_obj_sims[name_idx, other_idx] = new_obj_sims[other_idx, name_idx] = (new_objs[name] | new_objs[new_object_names[other_idx]]).item()

plot_similarity_matrix(new_obj_sims, new_object_names, values = True)