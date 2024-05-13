object_names = ['red','red^','red*circle','circle','circle^']

objs['red^'] = objs['red*circle'] * ~objs['circle']
objs['circle^'] = objs['red*circle'] * ~objs['red']