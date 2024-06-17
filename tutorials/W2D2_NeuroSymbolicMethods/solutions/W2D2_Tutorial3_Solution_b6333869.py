
sims = []

for obj_idx, obj in enumerate(obj_names):
    sims.append(query_ssps @ ssps[obj].flatten())