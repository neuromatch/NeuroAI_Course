set_seed(42)
vector_length = 1024

card_states = ['RED','BLUE','ODD','EVEN','NOT','GREEN','PRIME','IMPLIES','ANT','RELATION','CONS']
vocab = make_vocabulary(vector_length)
vocab.populate(';'.join(card_states))


for a in ['RED','BLUE','ODD','EVEN','GREEN','PRIME']:
    vocab.add(f'NOT_{a}', vocab['NOT'] * vocab[a])

action_names = ['RED','BLUE','ODD','EVEN','GREEN','PRIME','NOT_RED','NOT_BLUE','NOT_ODD','NOT_EVEN','NOT_GREEN','NOT_PRIME']
action_space = np.array([vocab[x].v for x in action_names]).squeeze()