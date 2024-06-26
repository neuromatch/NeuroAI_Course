class Env(nn.Module):
    def __init__(self, arg):
        """
        Initializes the environment with given arguments.

        Inputs:
        - arg (object): An object containing initialization parameters.

        Outputs:
        - None
        """
        super().__init__()
        self.__dict__.update(arg.__dict__)

    def reset(self, target_position=None, gain=None):
        """
        Resets the environment to start a new trial.

        Inputs:
        - target_position (tensor, optional): The target position for the trial. If None, a position is sampled.
        - gain (tensor, optional): The joystick gain. If None, the default gain is used.

        Outputs:
        - initial_state (tensor): The initial state of the environment.
        """
        # sample target position
        self.target_position = target_position
        if target_position is None:
            target_rel_r = torch.sqrt(torch.zeros(1).uniform_(*self.initial_radius_range**2))
            target_rel_ang = torch.zeros(1).uniform_(*self.relative_angle_range)
            rel_phi = np.pi/2 - target_rel_ang
            target_x = target_rel_r * torch.cos(rel_phi)
            target_y = target_rel_r * torch.sin(rel_phi)
            self.target_position = torch.tensor([target_x, target_y]).view([-1, 1])
        self.target_position_obs = self.target_position.clone()

        # joystick gain
        self.gain = gain
        if gain is None:
            self.gain = self.process_gain_default

        # process noise std
        self.pro_noise_std = self.gain * self.pro_noise_std_

        return torch.tensor([0, 0, np.pi / 2, 0, 0]).view([-1, 1])  # return the initial state

    def forward(self, x, a, t):
        """
        Updates the state based on the current state, action, and time.

        Inputs:
        - x (tensor): The current state of the environment.
        - a (tensor): The action taken.
        - t (int): The current time step.

        Outputs:
        - next_x (tensor): The next state of the environment.
        - reached_target (bool): Whether the target has been reached.
        """
        if t == self.target_offT:
            self.target_position_obs *= 0  # make target invisible

        relative_dist = torch.dist(x[:2], self.target_position)
        reached_target = relative_dist < self.goal_radius
        next_x = self.dynamics(x, a.view(-1))  # update state based on environment dynamics

        return next_x, reached_target

    def dynamics(self, x, a):
        """
        Defines the environment dynamics.

        Inputs:
        - x (tensor): The current state of the environment.
        - a (tensor): The action taken.

        Outputs:
        - next_x (tensor): The next state of the environment.
        """
        # sample process noise
        eta = self.pro_noise_std * torch.randn(2)

        # there are five elements in the state
        px, py, heading_angle, lin_vel, ang_vel = torch.split(x.view(-1), 1)

        # update state: s_{t+1} = f(s_{t}, a_{t})
        px_ = px + lin_vel * torch.cos(heading_angle) * self.DT
        # Hint: Mimic how the x position is updated. The y position update is similar,
        # but it requires the sine of 'heading_angle' instead of the cosine.
        py_ = py + lin_vel * torch.sin(heading_angle) * self.DT
        heading_angle_ = heading_angle + ang_vel * self.DT
        lin_vel_ = self.gain[0] * a[0] + eta[0]
        # Hint: The variables 'self.gain', 'a', and 'eta' are two-dimensional.
        # The first dimension is for the linear component, and the second dimension is for the angular component.
        ang_vel_ = self.gain[1] * a[1] + eta[1]

        next_x = torch.stack([px_, py_, heading_angle_,
                              lin_vel_.reshape(1), ang_vel_.reshape(1)]).view([-1, 1])
        return next_x

    def is_stop(self, action):
        """
        Determines if the given action is a stop action.

        Inputs:
        - action (tensor): The action.

        Outputs:
        - stop (bool): Whether the action is a stop action.
        """
        stop = (action.abs() < self.TERMINAL_ACTION).all()
        return stop