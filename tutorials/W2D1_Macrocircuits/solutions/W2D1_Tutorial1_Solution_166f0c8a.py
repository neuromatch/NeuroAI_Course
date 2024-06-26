
"""
Discussion: 1. What is the minimum error achievable by an MLP on the generated problem?
2. What is the minimum error achievable by a 1-hidden-layer MLP?

1. This is a trick question! We generated the data ourselves; the teacher network is an MLP. In principle, a student network with the same architecture could learn the exact weights of the teacher and achieve exactly 0 error.
2. By the universal approximator theorem, we can approximate the teacher network arbitrarily well with a 1-hidden-layer MLP, as long as there is not limit on the number of hidden units. So the answer is technically 0. In practice, however, when fitting a complex function, for example a deep teacher network, the number of hidden units required for low error can be impractical.
"""