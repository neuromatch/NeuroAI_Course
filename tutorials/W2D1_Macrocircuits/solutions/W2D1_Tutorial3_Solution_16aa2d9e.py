
"""
Discussion: To ensure a fair comparison, the total number of trainable parameters is designed to be similar between the two architectures. How many trainable parameters are there in each architecture?

Taking into consideration that OBS_DIM = ACTION_DIM = TARGET_DIM = 2, and that for `LSTM` layer, the total number of parameters is 4(nm + n^2 + n) where m is the input dimension and n is the output dimension (as we have self-recurrence, thus n^2, projection from input to output, thus nm, and, finally, bias, thus n), we have:

- for holistic actor (LSTM + Linear projection): 4 * (6 * 220 + 220*220 + 220) + (220 * 2 + 220) = 200420.
- for modular actor (LSTM + Linear projections): 4 * (6 * 128 + 128*128 + 128) + (128 * 300 + 128) + (300 * 300 + 300) + (300 * 2 + 300) = 198848.
"""