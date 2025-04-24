"""
The model does a pretty good job. However, we can see some mistakes in its transcription of the
first image, in particular, it fails to recognize the words "Neuromatch" and "Neuro AI".
This is likely due to the fact that the model's decoder was trained in 2019, prior to
the inception of Neuromatch in 2020 and the recent popularity of Neuro AI. Although it
has the capacity to express strings like "Neuromatch" and "Neuro AI", it assigns low
probabilities to these words, which weren't in its corpus. This is a clear example of
the importance of training data in building successful models.
"""