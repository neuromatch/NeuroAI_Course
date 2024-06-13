def evaluation(agent, gain_factor=1):
    """
    Evaluates the agent's performance in the environment.

    Inputs:
    - agent (Agent): The agent to be evaluated.
    - gain_factor (float): A factor to scale the process gain. Default is 1.

    Outputs:
    - results (DataFrame): A DataFrame containing evaluation results with columns:
      - 'pos_x': List of x positions over episodes.
      - 'pos_y': List of y positions over episodes.
      - 'pos_r_end': List of final radial distances from origin over episodes.
      - 'target_x': List of target x positions.
      - 'target_y': List of target y positions.
      - 'target_r': List of target radial distances.
      - 'rewarded': List of binary rewards indicating if the target was reached and stopped.
      - 'state_input': List of state inputs recorded during the episodes.
    """
    set_seed(0)
    env = Env(arg)

    pos_x = []; pos_y = []; pos_r_end = []
    target_x = []; target_y = []; target_r = []
    rewarded = []; state_input_ = []

    for target_position in tqdm(target_positions):
        state = env.reset(target_position=target_position, gain=arg.process_gain_default * gain_factor)
        agent.obs_step.reset(env.gain)

        state_input = torch.cat([torch.zeros([1, 1, arg.OBS_DIM]), torch.zeros([1, 1, arg.ACTION_DIM]),
                                 env.target_position_obs.view(1, 1, -1)], dim=2)
        hidden_in = (torch.zeros(1, 1, agent.actor.RNN_SIZE), torch.zeros(1, 1, agent.actor.RNN_SIZE))

        state_inputs = []
        states = []

        for t in range(arg.EPISODE_LEN):
            # 1. Agent takes an action given the state-related input
            action, hidden_out = agent.select_action(state_input, hidden_in)

            # 2. Environment updates to the next state given state and action,
            #    as well as checking if the agent has reached the reward zone,
            #    and if the agent has stopped.
            next_state, reached_target = env(state, action, t)
            is_stop = env.is_stop(action)

            # 3. Receive reward
            # TODO: Compute the reward. The reward is '1' when the target is reached and the agent stops on it,
            # otherwise, the reward is '0'.
            # Hint: Use variables 'reached_target' and 'is_stop'.
            reward = reached_target & is_stop

            # 4. Agent observes the next state and constructs the next state-related input
            next_observation = agent.obs_step(next_state)
            next_state_input = torch.cat([next_observation.view(1, 1, -1), action,
                                          env.target_position_obs.view(1, 1, -1)], dim=2)

            states.append(state)
            state_inputs.append(state_input)

            state_input = next_state_input
            state = next_state
            hidden_in = hidden_out

            # trial is done when the agent stops
            if is_stop:
                break

        # store data for each trial
        pos_x_, pos_y_, _, _, _ = torch.chunk(torch.cat(states, dim=1), state.shape[0], dim=0)
        pos_x.append(pos_x_.view(-1).numpy() * arg.LINEAR_SCALE)
        pos_y.append(pos_y_.view(-1).numpy() * arg.LINEAR_SCALE)

        pos_r, _ = cart2pol(pos_x[-1], pos_y[-1])
        pos_r_end.append(pos_r[-1])

        target_x.append(target_position[0].item() * arg.LINEAR_SCALE)
        target_y.append(target_position[1].item() * arg.LINEAR_SCALE)
        target_r_, _ = cart2pol(target_x[-1], target_y[-1])
        target_r.append(target_r_)

        state_input_.append(torch.cat(state_inputs))
        rewarded.append(reward.item())

    return(pd.DataFrame().assign(pos_x=pos_x, pos_y=pos_y,
                                 pos_r_end=pos_r_end, target_x=target_x, target_y=target_y,
                                 target_r=target_r, rewarded=rewarded,
                                 state_input=state_input_))