#unifying bundled representation of all objects
all_objs = (objs['circle'] + objs['square'] + objs['triangle']).normalize()

#unbind this unifying representation from the map
query_map = ssp_map * ~all_objs