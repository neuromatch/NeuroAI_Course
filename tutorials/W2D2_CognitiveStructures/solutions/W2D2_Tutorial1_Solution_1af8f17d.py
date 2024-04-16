
query_objs = np.vstack([objs[n] for n in obj_names])
test_positions = np.vstack((positions, [0,0], [0,-1.5]))

sims = []

for pos_idx, pos in enumerate(test_positions):
    position_ssp = ssp_space.encode(pos[None,:]) #remember we need to have 2-dimensional vectors for `encode()` function
    query_map = ssp_map * ~position_ssp
    sims.append(np.einsum('nd,d->n', query_objs,query_map.flatten()))