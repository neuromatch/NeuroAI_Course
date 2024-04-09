
def evaluate_individual(env, agent, optimizer_func, num_tasks = 10, num_gradient_steps = 25, num_trials = 6, num_evaluation_trials = 20):
    """Training and evaluation for agent in Harlow experiment environment for the bunch of tasks (thus measuring overall potential for agent's generalization across the tasks).
    Evaluation goes only after all gradient steps.

    Inputs:
    - env (HarlowExperimentEnv): environment.
    - agent (ActorCritic): particular instance of Actor Critic agent to train.
    - optimizer_func (torch.Optim): optimizer to use for training.
    - num_tasks (int, default = 10): number of tasks to evaluate agent on.
    - num_gradient_steps (int, default = 25): number of gradient steps to perform.
    - num_trials (int, default = 6): number of times the agent is exposed to the environment per gradient step to be trained .
    - num_evaluation_trials (int, default = 20): number of times the agent is exposed to the environment to evaluate it (no training happend during this phase).

    Outputs:
    - score (int): total score.
    - scores (list): list of scores obtained during evaluation on the specific tasks.
    """
    scores = []
    for _ in range(num_tasks): #lines[2-3]; notice that environment resets inside `train_evaluate_agent`
      agent_copy = copy.deepcopy(agent) #line[4]; remember that we don't want to change agent's parameters!
      score = train_evaluate_agent(env, agent_copy, optimizer_func, num_gradient_steps, num_trials, num_evaluation_trials)
      scores.append(score)
    return np.sum(scores), scores