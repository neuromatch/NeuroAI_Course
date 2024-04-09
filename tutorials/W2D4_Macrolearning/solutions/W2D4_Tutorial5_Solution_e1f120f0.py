
class ChangingEnv():
    def __init__(self, mode = 1):
        """Initialize changing environment.

        Inputs:
        - mode (int, default = 1): defines mode of the enviornment. Should be only 1 or 2.
        """
        if mode not in [1, 2]:
            raise ValueError("Mode is out of allowed range. Please consider entering 1 or 2 as digit.")

        self.mode = mode
        self.colors = mode_colors[self.mode]
        self.update_state()

    def update_state(self):
        """Update state which depends on the mode of the environment."""
        self.first_color, self.second_color = np.random.choice(self.colors, 2, replace = False)
        self.color_state = np.array([self.first_color, self.second_color])
        self.state = np.array([color_names_values[self.first_color], color_names_values[self.second_color]])

    def reset(self, mode = 1):
        """Reset environment by updating its mode (colors to sample from). Set the first state in the given mode."""
        self.mode = mode
        self.colors = mode_colors[self.mode]
        self.update_state()
        return self.state

    def step(self, action):
        """Evaluate agent's perfromance, return reward, max reward (for tracking agent's performance) and next observation."""
        feedback = color_names_rewards[self.color_state[action]]
        max_feedback = np.max([color_names_rewards[self.color_state[action]], color_names_rewards[self.color_state[1 - action]]])
        self.update_state()
        return self.state, feedback, max_feedback