
sims = []

for obj_idx, obj in enumerate(obj_names):
    sims.append(np.einsum('nd,d->n', query_ssps, ssps[obj].flatten()))