
def update_population(env, optimizer_func, population, parents_population, best_score, new_generation_new_individuals = 5):
    """
    Updates population with new individuals which are the result of crossing over and mutation of two parents agents.
    Removes the same amount of random agents from the population.

    Inputs:
    - env (HarlowExperimentEnv): environment.
    - optimizer_func (torch.Optim): optimizer to use for training.
    - population (list): current population which consists of tuples (agent, score).
    - parents_population (list) : parents individuals (part of current population) for creating new individuals.
    - best_score (int): the best score for the individual in the population registered so far.
    - new_generation_new_individuals (int, default = 5): the number of individuals to create (and the old ones to remove).
    """

    #create new individuals
    new_individuals = []
    for _ in range(new_generation_new_individuals):
        agent1, agent2 = random.choices(parents_population, k = 2)
        new_agent = create_new_agent(agent1[0], agent2[0])
        score, _ = evaluate_individual(env, new_agent, optimizer_func)
        #evaluate whether best score has increased
        best_score = max(score, best_score)
        new_individuals.append((new_agent, score))

    #remove random old individuals
    for _ in range(new_generation_new_individuals):
        population.pop(random.randint(0, len(population) - 1))

    return population + new_individuals, best_score