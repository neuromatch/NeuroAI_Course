
class HarlowExperimentEnv():
    def __init__(self, reward = 1, punishment = -1):
        """Initialize Harlow Experiment environment."""
        self.reward = reward
        self.punishment = punishment

        self.rewarded_digit = -1
        self.punished_digit = -1

        self.state = np.array([self.rewarded_digit, self.punished_digit])

    def update_state(self):
        """Update state by selecting rewarded hand for random."""
        if np.random.rand() < 0.5:
            self.state = np.array([self.rewarded_digit, self.punished_digit])
        else:
            self.state = np.array([self.punished_digit, self.rewarded_digit])

    def reset(self):
        """Reset environment by updating new rewarded and punished digits as well as create current state of the world (tuple of observations)."""
        self.rewarded_digit, self.punished_digit = np.random.choice(10, 2, replace=False)
        self.update_state()
        return self.state

    def step(self, action):
        """Evaluate agent's perfromance, return reward and next observation."""
        if self.state[action] == self.rewarded_digit:
            feedback = self.reward
        else:
            feedback = self.punishment
        self.update_state()
        return self.state, feedback