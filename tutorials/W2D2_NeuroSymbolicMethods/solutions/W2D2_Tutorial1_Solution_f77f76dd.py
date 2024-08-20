new_object_names = ['red','red^','red*circle','circle','circle^']
new_objs = objs

new_objs['red^'] = new_objs['red*circle'] * ~new_objs['circle']
new_objs['circle^'] = new_objs['red*circle'] * ~new_objs['red']