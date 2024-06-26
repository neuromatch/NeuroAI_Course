with plt.xkcd():
    trial_idx = 21
    trial = modular_df.iloc[trial_idx]

    fig = plt.figure(figsize=(2.2, 1.7), dpi=200)
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])

    # plot trajectory
    px = trial.pos_x; py = trial.pos_y
    ax.plot(px, py, lw=lw, c=modular_c)

    # plot target
    target_x = trial.target_x; target_y = trial.target_y
    print(f'Target distance from the start location: {np.around(trial.target_r, 1)} cm')

    # Given target locations as trial.target_x and trial.target_y,
    # and stop locations as trial.pos_x[-1] and trial.pos_y[-1],
    # compute the Euclidean distance between the target and stop locations.
    distance_stoploc_to_target = np.sqrt((trial.target_x - trial.pos_x[-1])**2 + (trial.target_y - trial.pos_y[-1])**2)
    print(f'Target distance from the stop location: {np.around(distance_stoploc_to_target, 1)} cm')

    print(f'Steps taken: {px.size - 1}')

    reward_boundary_radius = arg.goal_radius * arg.LINEAR_SCALE
    target_color = reward_c if distance_stoploc_to_target < reward_boundary_radius else unreward_c

    cir1 = Circle(xy=[target_x, target_y], radius=reward_boundary_radius, alpha=0.4, color=target_color, lw=0)
    ax.add_patch(cir1)
    ax.scatter(target_x, target_y, c=target_color, s=5)

    # plot initial position
    ax.scatter(0, 0, c='k', s=20, marker='*')
    ax.text(10, -10, s='Start', fontsize=fontsize)

    fig.tight_layout(pad=0)