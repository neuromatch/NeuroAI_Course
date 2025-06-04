# unifying bundled representation of all objects
all_objs = (objs['circle'] + objs['square'] + objs['triangle']).normalize()

# unbind this unifying representation from the map
query_map = ssp_map * ~all_objs

sims = query_ssps @ query_map.flatten()
size = (dim0.size,dim1.size)

plot_unbinding_objects_map(sims, positions, query_xs, size)