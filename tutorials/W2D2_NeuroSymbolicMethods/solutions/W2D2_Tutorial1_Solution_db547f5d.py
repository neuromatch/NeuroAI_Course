object_names = ['RED','EST_RED','RED_CIRCLE','CIRCLE','EST_CIRCLE']

vocab.add('EST_RED', (vocab['RED_CIRCLE'] * ~vocab['CIRCLE']).normalized())
vocab.add('EST_CIRCLE', (vocab['RED_CIRCLE'] * ~vocab['RED']).normalized())