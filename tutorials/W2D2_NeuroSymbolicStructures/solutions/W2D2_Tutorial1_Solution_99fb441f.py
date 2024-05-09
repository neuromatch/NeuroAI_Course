
objs['king'] = (objs['monarch'] + objs['male']).normalize()
objs['queen'] = (objs['monarch'] + objs['female']).normalize()
objs['prince'] = (objs['heir'] + objs['male']).normalize()
objs['princess'] = (objs['heir'] + objs['female']).normalize()