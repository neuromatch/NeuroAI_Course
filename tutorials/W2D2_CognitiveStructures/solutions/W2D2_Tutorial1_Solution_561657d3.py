
sims = []

for obj_idx, obj_name in enumerate(obj_names):
    query_map = ssp_map * ~objs[obj_name] # Query the object name
    sims.append(np.einsum('nd,d->n', query_ssps,query_map.flatten()))