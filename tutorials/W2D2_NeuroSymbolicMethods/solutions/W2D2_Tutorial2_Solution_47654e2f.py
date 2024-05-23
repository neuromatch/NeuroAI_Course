
set_seed(42)

card_states = ['red','blue','odd','even','not','green','prime','implies','ant','relation','cons']
encoder = sspspace.DiscreteSPSpace(card_states, ssp_dim=1024, optimize=False)
vocab = {c:encoder.encode(c) for c in card_states}

for a in ['red','blue','odd','even','green','prime']:
    vocab[f'not*{a}'] = vocab['not'] * vocab[a]

action_names = ['red','blue','odd','even','green','prime','not*red','not*blue','not*odd','not*even','not*green','not*prime']
action_space = np.array([vocab[x] for x in action_names]).squeeze()