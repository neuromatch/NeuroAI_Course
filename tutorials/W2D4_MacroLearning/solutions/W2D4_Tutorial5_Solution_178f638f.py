
class ReplayBuffer():
    def __init__(self, max_experience = 250, num_trials = 100):
        """Initialize replay buffer.
        Notice that when replay buffer is full of experience and new one should be remembered, it replaces existing ones, starting
        from the oldest.

        Inputs:
        - max_experience (int, default = 250): the maximum number of experience (gradient steps) which can be stored.
        - num_trials (int, default = 100): number of times the agent is exposed to the environment per gradient step to be trained.
        """
        self.max_experience = max_experience

        #variable which fully describe experience
        self.losses = [0 for _ in range(self.max_experience)]

        #number of memory cell to point to (write or overwrite experience)
        self.writing_pointer = 0
        self.reading_pointer = 0

        #to keep track how many experience there were
        self.num_experience = 0

    def write_experience(self, loss):
        """Write new experience."""
        self.losses[self.writing_pointer] = loss

        #so that pointer is in range of max_experience and will point to the older experience while full
        self.writing_pointer = (self.writing_pointer + 1) % self.max_experience
        self.num_experience += 1

    def read_experience(self):
        """Read existing experience."""
        loss = self.losses[self.reading_pointer]

        #so that pointer is in range of self.max_experience and will point to the older experience while full
        self.reading_pointer = (self.reading_pointer + 1) % min(self.max_experience, self.num_experience)
        return loss