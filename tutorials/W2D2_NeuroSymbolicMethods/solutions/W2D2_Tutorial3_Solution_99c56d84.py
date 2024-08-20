
objects_sims = []

for obj_idx, obj_name in enumerate(obj_names):
    #query the object name by unbinding it from the map
    query_map = ssp_map * ~objs[obj_name]
    objects_sims.append(query_ssps @ query_map.flatten())

plot_2d_similarity(objects_sims, obj_names, (dim0.size, dim1.size), title_argmax = True)