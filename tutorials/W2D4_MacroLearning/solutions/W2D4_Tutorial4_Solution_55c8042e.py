
def create_initial_population(env, optmizer_func, population_size = 50, hidden_size = 20):
    """
    Creates an initial population of agents.

    Inputs:
    - env (HarlowExperimentEnv): environment.
    - optimizer_func (torch.Optim): optimizer to use for training.
    - population_size (int, default = 50): the size of the initial population.
    - hidden_size (int, default = 20): the size of LSTM layer in A2C agent.

    Outputs:
    - population (list): initial population which consists of tuples (agent, score).
    - best_score (int): the best score for the individual in the population registered so far.
    """
    population = []
    total_score = 0
    best_score = 0
    for _ in range(population_size):
        agent = ActorCritic(hidden_size)
        score, _ = evaluate_individual(env, agent, optmizer_func)
        best_score = max(best_score, score)
        total_score += score
        population.append((agent, score))
    print(f"Generation: 0, mean population score: {total_score / population_size}, best score: {best_score}")
    return population, best_score

def create_new_agent(agent1, agent2):
    """
    Creates new agent using crossing over technique over layers of network and mutation of the parameters with Gaussian noise.

    Inputs:
    - agent1 (ActorCritic): first parent agent.
    - agent2 (ActorCritic): second parent agent.

    Outputs:
    - new_agent (ActorCritic): new agent which is offspring of the given two.
    """
    #creates agent as copy of the first one
    new_agent = copy.deepcopy(agent1)

    #evolving network parameters with crossing over (over separate layes) & mutating (Gaussian noise)
    for name, module in new_agent.named_modules():
        if isinstance(module, nn.Linear):
            if random.random() < 0.5:
                module.weight.data = agent2._modules[name].weight.data
                module.bias.data = agent2._modules[name].bias.data
            #add noise
            module.weight.data += torch.randn_like(module.weight.data) * parameters_noise
            module.bias.data += torch.randn_like(module.bias.data) * parameters_noise
        elif isinstance(module, nn.LSTM):
            if random.random() < 0.5:
                module.weight_ih_l0.data = agent2._modules[name].weight_ih_l0.data
                module.weight_hh_l0.data = agent2._modules[name].weight_hh_l0.data
                module.bias_ih_l0.data = agent2._modules[name].bias_ih_l0.data
                module.bias_hh_l0.data = agent2._modules[name].bias_hh_l0.data
            #add noise
            module.weight_ih_l0.data += torch.randn_like(module.weight_ih_l0.data) * parameters_noise
            module.weight_hh_l0.data += torch.randn_like(module.weight_hh_l0.data) * parameters_noise
            module.bias_ih_l0.data += torch.randn_like(module.bias_ih_l0.data) * parameters_noise
            module.bias_hh_l0.data += torch.randn_like(module.bias_hh_l0.data) * parameters_noise

    #evolving & mutating hyperparameters
    if random.random() < 0.5:
        new_agent.learning_rate = agent2.learning_rate
    new_agent.learning_rate += np.random.normal(size = 1).item() * learning_rate_noise
    new_agent.learning_rate = min(max(new_agent.learning_rate, 0.0001), 0.01)
    if random.random() < 0.5:
        new_agent.discount_factor = agent2.discount_factor
    new_agent.discount_factor += np.random.normal(size = 1).item() * discount_factor_noise
    new_agent.discount_factor = min(max(new_agent.discount_factor, 0.6), 0.99)
    if random.random() < 0.5:
        new_agent.state_value_estimate_cost = agent2.state_value_estimate_cost
    new_agent.state_value_estimate_cost += np.random.normal(size = 1).item() * state_value_estimate_cost_noise
    new_agent.state_value_estimate_cost = min(max(new_agent.discount_factor, 0.1), 0.7)
    if random.random() < 0.5:
        new_agent.entropy_cost = agent2.entropy_cost
    new_agent.entropy_cost += np.random.normal(size = 1).item() * entropy_cost_noise
    new_agent.entropy_cost = min(max(new_agent.discount_factor, 0.0001), 0.05)

    return new_agent

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